-- Supabase → SQL Editor (executar uma vez).
-- Tabela em snake_case (corresponde ao python3 -m src.upload_supabase).

create table if not exists public.diabetes (
  pregnancies double precision not null,
  glucose double precision not null,
  blood_pressure double precision not null,
  skin_thickness double precision not null,
  insulin double precision not null,
  bmi double precision not null,
  diabetes_pedigree_function double precision not null,
  age double precision not null,
  outcome integer not null
);

-- Desenvolvimento: sem RLS (em produção ativa RLS e políticas adequadas).
alter table public.diabetes disable row level security;
