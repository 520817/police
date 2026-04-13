import psycopg2
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def get_db_connection():
    try:
        db_url = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        print(f"Connection Error: {e}")
        return None


def _get_table_columns(cur, table_name: str):
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table_name,),
    )
    return {row[0] for row in cur.fetchall()}


def save_survey_scores(
    session_id,
    score_type,
    valence,
    arousal,
    vas,
    validation_q1_text=None,
    validation_q2_text=None,
    validation_q3_text=None,
    validation_q1=None,
    validation_q2=None,
    validation_q3=None,
):
    if not session_id or session_id == "null":
        print("[DB Error] session_id가 유효하지 않습니다.")
        return

    conn = get_db_connection()
    if not conn:
        return
    cur = conn.cursor()

    user_prt = session_id.split("_")[0]

    try:
        session_columns = _get_table_columns(cur, "sessions")

        if score_type == "pre":
            sql = """
            INSERT INTO sessions (session_id, prt, pre_valence, pre_arousal, pre_vas, pre_timestamp)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (session_id)
            DO UPDATE SET
                pre_valence = EXCLUDED.pre_valence,
                pre_arousal = EXCLUDED.pre_arousal,
                pre_vas = EXCLUDED.pre_vas,
                pre_timestamp = CURRENT_TIMESTAMP;
            """
            cur.execute(sql, (session_id, user_prt, int(valence), int(arousal), int(vas)))
        else:
            q1 = int(validation_q1) if validation_q1 is not None else 0
            q2 = int(validation_q2) if validation_q2 is not None else 0
            q3 = int(validation_q3) if validation_q3 is not None else 0

            candidate_updates = [
                ("post_validation_q1_text", validation_q1_text),
                ("post_validation_q2_text", validation_q2_text),
                ("post_validation_q3_text", validation_q3_text),
                ("post_valence", int(valence)),
                ("post_arousal", int(arousal)),
                ("post_vas", int(vas)),
                ("post_validation_q1", q1),
                ("post_validation_q2", q2),
                ("post_validation_q3", q3),
            ]

            assignments = []
            params = []
            for column_name, column_value in candidate_updates:
                if column_name in session_columns:
                    assignments.append(f"{column_name}=%s")
                    params.append(column_value)

            if "post_timestamp" in session_columns:
                assignments.append("post_timestamp=CURRENT_TIMESTAMP")

            if not assignments:
                raise RuntimeError("sessions 테이블에 저장 가능한 post 설문 컬럼이 없습니다.")

            sql = f"""
            UPDATE sessions
            SET {", ".join(assignments)}
            WHERE session_id=%s
            """
            params.append(session_id)
            cur.execute(sql, tuple(params))

            if cur.rowcount == 0:
                raise RuntimeError(f"post survey update 대상 session_id를 찾지 못했습니다: {session_id}")

        conn.commit()
        print(f"Survey {score_type} saved for {session_id}")
    except Exception as e:
        print(f"DB Error in save_survey_scores: {e}")
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def save_biosignal_log(session_id, biosignal_summary, valid_record_count):
    conn = get_db_connection()
    if not conn:
        return
    cur = conn.cursor()

    sql = """
    INSERT INTO biosignal_logs (session_id, biosignal_summary, valid_record_count)
    VALUES (%s, %s, %s)
    """
    try:
        cur.execute(sql, (session_id, biosignal_summary, valid_record_count))
        conn.commit()
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def save_chat_message(session_id, role, content, situation=None, emotion=None, stage=None, thought=None):
    conn = get_db_connection()
    if not conn:
        return
    cur = conn.cursor()
    try:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (session_id,))

        sql_order = """
        SELECT COALESCE(MAX(message_order), 0) + 1
        FROM chat_messages
        WHERE session_id = %s
        """
        cur.execute(sql_order, (session_id,))
        message_order = cur.fetchone()[0]

        sql = """
        INSERT INTO chat_messages (session_id, message_order, role, content, situation, emotion, stage, thought)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(sql, (session_id, message_order, role, content, situation, emotion, stage, thought))
        conn.commit()
    except Exception as e:
        print(f"DB Error in save_chat_message: {e}")
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()
