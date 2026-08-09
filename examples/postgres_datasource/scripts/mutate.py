"""Deterministic source mutations for incremental ingestion demos."""

from __future__ import annotations

import asyncio
import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

SOURCE_URL = os.environ.get(
    "COGKURA_POSTGRES_SOURCE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/cogkura_source",
)


async def main() -> None:
    engine = create_async_engine(SOURCE_URL)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                INSERT INTO public.messages (
                    id, conversation_id, user_id, sender_type, body,
                    created_at, updated_at
                ) VALUES (
                    '99999999-9999-9999-9999-999999999999',
                    '22222222-2222-2222-2222-222222222222',
                    '11111111-1111-1111-1111-111111111111',
                    'user',
                    'George is evaluating async ingestion checkpoints.',
                    '2026-08-04T12:00:00+00:00',
                    '2026-08-04T12:00:00+00:00'
                )
                """
            )
        )
        await conn.execute(
            text(
                """
                UPDATE public.messages
                SET body = 'Draft release notes for v0.1 (revised).',
                    updated_at = '2026-08-04T12:01:00+00:00'
                WHERE id = '00000000-0000-0000-0000-000000000007'
                """
            )
        )
        await conn.execute(
            text(
                """
                UPDATE public.messages
                SET deleted_at = '2026-08-04T12:02:00+00:00',
                    updated_at = '2026-08-04T12:02:00+00:00'
                WHERE id = '00000000-0000-0000-0000-000000000004'
                """
            )
        )
        await conn.execute(
            text(
                """
                INSERT INTO public.messages (
                    id, conversation_id, user_id, sender_type, body,
                    created_at, updated_at
                ) VALUES (
                    '88888888-8888-8888-8888-888888888888',
                    '22222222-2222-2222-2222-222222222222',
                    '11111111-1111-1111-1111-111111111111',
                    'user',
                    'PostgreSQL remains George''s default database choice.',
                    '2026-08-04T12:03:00+00:00',
                    '2026-08-04T12:03:00+00:00'
                )
                """
            )
        )
        await conn.execute(
            text(
                """
                INSERT INTO public.messages (
                    id, conversation_id, user_id, sender_type, body,
                    created_at, updated_at
                ) VALUES (
                    '77777777-7777-7777-7777-777777777777',
                    '33333333-3333-3333-3333-333333333333',
                    '11111111-1111-1111-1111-111111111111',
                    'user',
                    'For this analytics rollout, AWS is required.',
                    '2026-08-04T12:04:00+00:00',
                    '2026-08-04T12:04:00+00:00'
                )
                """
            )
        )
    await engine.dispose()
    print("Source mutations applied.")


if __name__ == "__main__":
    asyncio.run(main())
