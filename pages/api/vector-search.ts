import type { NextRequest } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { codeBlock, oneLine } from 'common-tags'
import GPT3Tokenizer from 'gpt3-tokenizer'
import {
  Configuration,
  OpenAIApi,
  CreateModerationResponse,
  CreateEmbeddingResponse,
  ChatCompletionRequestMessage,
} from 'openai-edge'
import { OpenAIStream, StreamingTextResponse } from 'ai'
import { ApplicationError, UserError } from '@/lib/errors'

// Server-only credentials. The secret key is never sent to the browser.
const openAiKey = process.env.OPENAI_API_KEY || process.env.OPENAI_KEY
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseSecretKey = process.env.SUPABASE_SECRET_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY

const config = new Configuration({
  apiKey: openAiKey,
})
const openai = new OpenAIApi(config)

export default async function handler(req: NextRequest) {
  try {
    if (req.method !== 'POST') {
      return new Response(JSON.stringify({ error: 'Method not allowed' }), {
        status: 405,
        headers: { 'Content-Type': 'application/json', Allow: 'POST' },
      })
    }

    if (!openAiKey) {
      throw new ApplicationError('Missing environment variable OPENAI_API_KEY')
    }

    if (!supabaseUrl) {
      throw new ApplicationError('Missing environment variable NEXT_PUBLIC_SUPABASE_URL')
    }

    if (!supabaseSecretKey) {
      throw new ApplicationError('Missing server-only Supabase secret key')
    }

    const requestData = await req.json()
    if (!requestData) {
      throw new UserError('Missing request data')
    }

    const query = typeof requestData.prompt === 'string' ? requestData.prompt.trim() : ''
    if (!query) {
      throw new UserError('Missing query in request data')
    }

    if (query.length > 4000) {
      throw new UserError('Query is too long')
    }

    const supabaseClient = createClient(supabaseUrl, supabaseSecretKey, {
      auth: {
        persistSession: false,
        autoRefreshToken: false,
      },
    })

    const moderationResponse: CreateModerationResponse = await openai
      .createModeration({ input: query })
      .then((res) => res.json())

    const [results] = moderationResponse.results || []
    if (results?.flagged) {
      throw new UserError('Flagged content', {
        flagged: true,
        categories: results.categories,
      })
    }

    const embeddingResponse = await openai.createEmbedding({
      model: 'text-embedding-3-small',
      input: query.replace(/\n/g, ' '),
    })

    if (!embeddingResponse.ok) {
      throw new ApplicationError('Failed to create embedding for question', await embeddingResponse.text())
    }

    const {
      data: [{ embedding }],
    }: CreateEmbeddingResponse = await embeddingResponse.json()

    const { error: matchError, data: pageSections } = await supabaseClient.rpc(
      'match_page_sections',
      {
        embedding,
        match_threshold: 0.78,
        match_count: 10,
        min_content_length: 50,
      }
    )

    if (matchError) {
      throw new ApplicationError('Failed to match page sections', matchError)
    }

    if (!pageSections?.length) {
      throw new UserError('No matching documentation was found')
    }

    const tokenizer = new GPT3Tokenizer({ type: 'gpt3' })
    let tokenCount = 0
    let contextText = ''

    for (const pageSection of pageSections) {
      const content = pageSection.content || ''
      const encoded = tokenizer.encode(content)
      tokenCount += encoded.text.length

      if (tokenCount >= 1500) break
      contextText += `${content.trim()}\n---\n`
    }

    const prompt = codeBlock`
      ${oneLine`
        You are a very enthusiastic Supabase representative who loves
        to help people! Given the following sections from the Supabase
        documentation, answer the question using only that information,
        outputted in markdown format. If you are unsure and the answer
        is not explicitly written in the documentation, say
        "Sorry, I don't know how to help with that."
      `}

      Context sections:
      ${contextText}

      Question: """
      ${query}
      """

      Answer as markdown (including related code snippets if available):
    `

    const chatMessage: ChatCompletionRequestMessage = {
      role: 'user',
      content: prompt,
    }

    const response = await openai.createChatCompletion({
      model: 'gpt-4o-mini',
      messages: [chatMessage],
      max_tokens: 512,
      temperature: 0,
      stream: true,
    })

    if (!response.ok) {
      throw new ApplicationError('Failed to generate completion', await response.text())
    }

    return new StreamingTextResponse(OpenAIStream(response))
  } catch (err: unknown) {
    if (err instanceof UserError) {
      return new Response(
        JSON.stringify({ error: err.message, data: err.data }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    if (err instanceof ApplicationError) {
      console.error(`${err.message}: ${JSON.stringify(err.data)}`)
    } else {
      console.error(err)
    }

    return new Response(
      JSON.stringify({ error: 'There was an error processing your request' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
