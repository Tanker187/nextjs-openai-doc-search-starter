import { createClient } from '@supabase/supabase-js'
import { createHash } from 'crypto'
import dotenv from 'dotenv'
import { ObjectExpression } from 'estree'
import { readdir, readFile, stat } from 'fs/promises'
import GithubSlugger from 'github-slugger'
import { Content, Root } from 'mdast'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { mdxFromMarkdown, MdxjsEsm } from 'mdast-util-mdx'
import { toMarkdown } from 'mdast-util-to-markdown'
import { toString } from 'mdast-util-to-string'
import { mdxjs } from 'micromark-extension-mdxjs'
import { Configuration, OpenAIApi } from 'openai'
import { basename, dirname, join } from 'path'
import { u } from 'unist-builder'
import { filter } from 'unist-util-filter'
import { inspect } from 'util'
import yargs from 'yargs'

dotenv.config()

const ignoredFiles = ['pages/404.mdx']

function getObjectFromExpression(node: ObjectExpression) {
  return node.properties.reduce<
    Record<string, string | number | bigint | true | RegExp | undefined>
  >((object, property) => {
    if (property.type !== 'Property') return object

    const key = (property.key.type === 'Identifier' && property.key.name) || undefined
    const value = (property.value.type === 'Literal' && property.value.value) || undefined
    if (!key) return object

    return { ...object, [key]: value }
  }, {})
}

function extractMetaExport(mdxTree: Root) {
  const metaExportNode = mdxTree.children.find((node): node is MdxjsEsm => {
    return (
      node.type === 'mdxjsEsm' &&
      node.data?.estree?.body[0]?.type === 'ExportNamedDeclaration' &&
      node.data.estree.body[0].declaration?.type === 'VariableDeclaration' &&
      node.data.estree.body[0].declaration.declarations[0]?.id.type === 'Identifier' &&
      node.data.estree.body[0].declaration.declarations[0].id.name === 'meta'
    )
  })

  if (!metaExportNode) return undefined

  const objectExpression =
    (metaExportNode.data?.estree?.body[0]?.type === 'ExportNamedDeclaration' &&
      metaExportNode.data.estree.body[0].declaration?.type === 'VariableDeclaration' &&
      metaExportNode.data.estree.body[0].declaration.declarations[0]?.id.type === 'Identifier' &&
      metaExportNode.data.estree.body[0].declaration.declarations[0].id.name === 'meta' &&
      metaExportNode.data.estree.body[0].declaration.declarations[0].init?.type ===
        'ObjectExpression' &&
      metaExportNode.data.estree.body[0].declaration.declarations[0].init) ||
    undefined

  if (!objectExpression) return undefined
  return getObjectFromExpression(objectExpression)
}

function splitTreeBy(tree: Root, predicate: (node: Content) => boolean) {
  return tree.children.reduce<Root[]>((trees, node) => {
    const [lastTree] = trees.slice(-1)
    if (!lastTree || predicate(node)) return trees.concat(u('root', [node]))
    lastTree.children.push(node)
    return trees
  }, [])
}

type Meta = ReturnType<typeof extractMetaExport>
type Section = { content: string; heading?: string; slug?: string }
type ProcessedMdx = { checksum: string; meta: Meta; sections: Section[] }

function processMdxForSearch(content: string): ProcessedMdx {
  const checksum = createHash('sha256').update(content).digest('base64')
  const mdxTree = fromMarkdown(content, {
    extensions: [mdxjs()],
    mdastExtensions: [mdxFromMarkdown()],
  })
  const meta = extractMetaExport(mdxTree)
  const mdTree = filter(mdxTree, (node) =>
    !['mdxjsEsm', 'mdxJsxFlowElement', 'mdxJsxTextElement', 'mdxFlowExpression', 'mdxTextExpression'].includes(node.type)
  )

  if (!mdTree) return { checksum, meta, sections: [] }

  const sectionTrees = splitTreeBy(mdTree, (node) => node.type === 'heading')
  const slugger = new GithubSlugger()
  const sections = sectionTrees.map((tree) => {
    const [firstNode] = tree.children
    const heading = firstNode.type === 'heading' ? toString(firstNode) : undefined
    return { content: toMarkdown(tree), heading, slug: heading ? slugger.slug(heading) : undefined }
  })

  return { checksum, meta, sections }
}

type WalkEntry = { path: string; parentPath?: string }

async function walk(dir: string, parentPath?: string): Promise<WalkEntry[]> {
  const immediateFiles = await readdir(dir)
  const recursiveFiles = await Promise.all(
    immediateFiles.map(async (file) => {
      const path = join(dir, file)
      const stats = await stat(path)
      if (stats.isDirectory()) {
        const docPath = `${basename(path)}.mdx`
        return walk(path, immediateFiles.includes(docPath) ? join(dirname(path), docPath) : parentPath)
      }
      if (stats.isFile()) return [{ path, parentPath }]
      return []
    })
  )
  return recursiveFiles.reduce((all, folderContents) => all.concat(folderContents), []).sort((a, b) => a.path.localeCompare(b.path))
}

class MarkdownEmbeddingSource {
  type: 'markdown' = 'markdown'
  checksum?: string
  meta?: Meta
  sections?: Section[]

  constructor(public source: string, public filePath: string, public parentFilePath?: string) {}

  get path() { return this.filePath.replace(/^pages/, '').replace(/\.mdx?$/, '') }
  get parentPath() { return this.parentFilePath?.replace(/^pages/, '').replace(/\.mdx?$/, '') }

  async load() {
    const contents = await readFile(this.filePath, 'utf8')
    const { checksum, meta, sections } = processMdxForSearch(contents)
    this.checksum = checksum
    this.meta = meta
    this.sections = sections
    return { checksum, meta, sections }
  }
}

async function generateEmbeddings() {
  const argv = await yargs.option('refresh', {
    alias: 'r', description: 'Refresh data', type: 'boolean',
  }).argv
  const shouldRefresh = argv.refresh
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseSecretKey = process.env.SUPABASE_SECRET_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY
  const openAiKey = process.env.OPENAI_API_KEY || process.env.OPENAI_KEY

  if (!supabaseUrl || !supabaseSecretKey || !openAiKey) {
    return console.log('NEXT_PUBLIC_SUPABASE_URL, a server-only Supabase secret key, and OPENAI_API_KEY are required: skipping embeddings generation')
  }

  const supabaseClient = createClient(supabaseUrl, supabaseSecretKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  })
  const openai = new OpenAIApi(new Configuration({ apiKey: openAiKey }))

  const embeddingSources = (await walk('pages'))
    .filter(({ path }) => /\.mdx?$/.test(path))
    .filter(({ path }) => !ignoredFiles.includes(path))
    .map((entry) => new MarkdownEmbeddingSource('guide', entry.path, entry.parentPath))

  console.log(`Discovered ${embeddingSources.length} pages`)

  for (const embeddingSource of embeddingSources) {
    const { type, source, path, parentPath } = embeddingSource
    try {
      const { checksum, meta, sections } = await embeddingSource.load()
      const { error: fetchPageError, data: existingPage } = await supabaseClient
        .from('nods_page').select('id, path, checksum, parentPage:parent_page_id(id, path)')
        .filter('path', 'eq', path).limit(1).maybeSingle()
      if (fetchPageError) throw fetchPageError

      if (!shouldRefresh && existingPage?.checksum === checksum) continue

      if (existingPage) {
        const { error } = await supabaseClient.from('nods_page_section').delete().filter('page_id', 'eq', existingPage.id)
        if (error) throw error
      }

      const { data: parentPage, error: parentError } = await supabaseClient
        .from('nods_page').select().filter('path', 'eq', parentPath).limit(1).maybeSingle()
      if (parentError) throw parentError

      const { error: upsertError, data: page } = await supabaseClient
        .from('nods_page').upsert({ checksum: null, path, type, source, meta, parent_page_id: parentPage?.id }, { onConflict: 'path' })
        .select().limit(1).single()
      if (upsertError) throw upsertError

      for (const { slug, heading, content } of sections) {
        const input = content.replace(/\n/g, ' ')
        const embeddingResponse = await openai.createEmbedding({ model: 'text-embedding-3-small', input })
        if (embeddingResponse.status !== 200) throw new Error(inspect(embeddingResponse.data, false, 2))
        const [responseData] = embeddingResponse.data.data
        const { error } = await supabaseClient.from('nods_page_section').insert({
          page_id: page.id, slug, heading, content,
          token_count: embeddingResponse.data.usage.total_tokens,
          embedding: responseData.embedding,
        })
        if (error) throw error
      }

      const { error: updateError } = await supabaseClient.from('nods_page').update({ checksum }).filter('id', 'eq', page.id)
      if (updateError) throw updateError
    } catch (err) {
      console.error(`Failed to index page '${path}'`, err)
    }
  }

  console.log('Embedding generation complete')
}

generateEmbeddings().catch((err) => console.error(err))
