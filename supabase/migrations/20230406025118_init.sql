-- Enable pgvector for document embeddings.
create extension if not exists vector with schema public;

create table if not exists public.nods_page (
  id bigserial primary key,
  parent_page_id bigint references public.nods_page,
  path text not null unique,
  checksum text,
  meta jsonb,
  type text,
  source text
);

create table if not exists public.nods_page_section (
  id bigserial primary key,
  page_id bigint not null references public.nods_page on delete cascade,
  content text,
  token_count int,
  embedding vector(1536),
  slug text,
  heading text
);

alter table public.nods_page enable row level security;
alter table public.nods_page_section enable row level security;

-- The application uses a server-side Supabase secret for indexing and search.
-- No public table policies are granted here, so browser clients cannot directly
-- read or mutate the documentation tables.

create or replace function public.match_page_sections(
  embedding vector(1536),
  match_threshold float,
  match_count int,
  min_content_length int
)
returns table (
  id bigint,
  page_id bigint,
  slug text,
  heading text,
  content text,
  similarity float
)
language plpgsql
security invoker
set search_path = public
as $$
#variable_conflict use_variable
begin
  return query
  select
    s.id,
    s.page_id,
    s.slug,
    s.heading,
    s.content,
    (s.embedding <#> embedding) * -1 as similarity
  from public.nods_page_section as s
  where length(s.content) >= min_content_length
    and (s.embedding <#> embedding) * -1 > match_threshold
  order by s.embedding <#> embedding
  limit match_count;
end;
$$;

create or replace function public.get_page_parents(page_id bigint)
returns table (
  id bigint,
  parent_page_id bigint,
  path text,
  meta jsonb
)
language sql
security invoker
set search_path = public
as $$
  with recursive chain as (
    select *
    from public.nods_page
    where nods_page.id = page_id

    union all

    select child.*
    from public.nods_page as child
    join chain on chain.parent_page_id = child.id
  )
  select id, parent_page_id, path, meta
  from chain;
$$;
