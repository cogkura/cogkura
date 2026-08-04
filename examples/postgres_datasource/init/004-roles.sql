\c cognema_source
CREATE ROLE cognema_reader LOGIN PASSWORD 'cognema_reader';
GRANT CONNECT ON DATABASE cognema_source TO cognema_reader;
GRANT USAGE ON SCHEMA public TO cognema_reader;
GRANT SELECT ON public.users, public.conversations, public.messages TO cognema_reader;

\c cognema_memory
CREATE ROLE cognema_writer LOGIN PASSWORD 'cognema_writer';
GRANT CONNECT ON DATABASE cognema_memory TO cognema_writer;
GRANT USAGE ON SCHEMA cognema TO cognema_writer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA cognema TO cognema_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA cognema
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO cognema_writer;
