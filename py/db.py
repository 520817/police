import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    try:
        db_url = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(db_url, options="-c timezone=Asia/Seoul")
        return conn
    except Exception as e:
        print(f"Connection Error: {e}")
        return None


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
    is_insufficient=None,
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
        if score_type == "pre":
            # sessions에 메타 정보만 upsert (survey 컬럼 없음)
            cur.execute(
                """
                INSERT INTO sessions (session_id, prt)
                VALUES (%s, %s)
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id, user_prt),
            )

            # survey_log에 pre 매 제출마다 append
            cur.execute(
                """
                INSERT INTO survey_log (session_id, score_type, valence, arousal, vas)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, "pre", int(valence), int(arousal), int(vas)),
            )

        else:
            q1 = int(validation_q1) if validation_q1 is not None else None
            q2 = int(validation_q2) if validation_q2 is not None else None
            q3 = int(validation_q3) if validation_q3 is not None else None

            # survey_log에 post append
            cur.execute(
                """
                INSERT INTO survey_log (
                    session_id, score_type, valence, arousal, vas,
                    validation_q1, validation_q2, validation_q3,
                    validation_q1_text, validation_q2_text, validation_q3_text,
                    is_insufficient
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id, "post", int(valence), int(arousal), int(vas),
                    q1, q2, q3,
                    validation_q1_text, validation_q2_text, validation_q3_text,
                    is_insufficient,
                ),
            )

        conn.commit()
        print(f"Survey {score_type} saved for {session_id}")
    except Exception as e:
        print(f"DB Error in save_survey_scores: {e}")
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def save_biosignal_log(session_id, biosignal_result, biosignal_summary, opening_question, valid_record_count, plot_path=None):
    conn = get_db_connection()
    if not conn:
        return
    cur = conn.cursor()

    sql = """
    INSERT INTO biosignal_logs (session_id, biosignal_result, biosignal_summary, opening_question, valid_record_count, plot_path)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        cur.execute(sql, (session_id, biosignal_result, biosignal_summary, opening_question, valid_record_count, plot_path))
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
