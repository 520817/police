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

def save_survey_scores(session_id, score_type, valence, arousal, vas, validation_q1_text=None, validation_q2_text=None, validation_q3_text=None, validation_q1=None, validation_q2=None, validation_q3=None,):
    if not session_id or session_id == "null":
        print("❌ [DB Error] session_id가 유효하지 않습니다.")
        return
    
    conn = get_db_connection()
    if not conn: return
    cur = conn.cursor()
    
    # prt 추출 (ID의 첫 번째 부분)
    user_prt = session_id.split('_')[0]

    try:
        if score_type == "pre":
            # pre는 해당 session_id가 없을 때 INSERT, 있으면 해당 컬럼들만 UPDATE (UPSERT 방식)
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
            # post는 반드시 기존 행이 있어야 함 (설계상 pre가 먼저 실행됨)
            # 만약 pre 없이 post만 올 경우를 대비해 여기서도 INSERT/UPDATE(UPSERT)를 쓸 수 있음
            sql = """
            UPDATE sessions 
            SET post_validation_q1_text=%s, post_validation_q2_text=%s, post_validation_q3_text=%s,
                post_valence=%s, post_arousal=%s, post_vas=%s, post_validation_q1 = %s,
                post_validation_q2 = %s, post_validation_q3 = %s, post_timestamp=CURRENT_TIMESTAMP 
            WHERE session_id=%s
            """
            q1 = int(validation_q1) if validation_q1 is not None else 0
            q2 = int(validation_q2) if validation_q2 is not None else 0
            q3 = int(validation_q3) if validation_q3 is not None else 0
            
            cur.execute(sql, (int(valence), int(arousal), validation_q1_text, validation_q2_text, validation_q3_text, int(vas), q1, q2, q3, session_id))
            
        conn.commit()
        print(f"Survey {score_type} saved for {session_id}")
    except Exception as e:
        print(f"DB Error in save_survey_scores: {e}")
        conn.rollback()
        raise e # main.py에서 에러를 인지할 수 있게 던짐
    finally:
        cur.close()
        conn.close()

def save_biosignal_log(session_id, biosignal_summary, valid_record_count):
    conn = get_db_connection()
    if not conn: return
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
        # 같은 session_id 안에서는 message_order를 직렬화해서 중복 순번을 막는다.
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
