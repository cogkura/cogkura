\c cogkura_source
CREATE ROLE cogkura_reader LOGIN PASSWORD 'cogkura_reader';
GRANT CONNECT ON DATABASE cogkura_source TO cogkura_reader;
GRANT USAGE ON SCHEMA public TO cogkura_reader;
GRANT SELECT ON public.users, public.conversations, public.messages TO cogkura_reader;

\c cogkura_memory
CREATE ROLE cogkura_writer LOGIN PASSWORD 'cogkura_writer';
GRANT CONNECT ON DATABASE cogkura_memory TO cogkura_writer;
GRANT USAGE ON SCHEMA cogkura TO cogkura_writer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA cogkura TO cogkura_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA cogkura
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO cogkura_writer;
