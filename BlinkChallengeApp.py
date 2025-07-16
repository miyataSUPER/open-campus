"""
@file BlinkChallengeApp.py
@brief まばたき回数チャレンジアプリ（Streamlit）

このスクリプトは、Webカメラを用いてユーザーのまばたき回数を30秒間計測し、リーダーボードに記録します。
MediaPipeで顔のランドマークを検出し、EAR（Eye Aspect Ratio）でまばたきを判定します。

主な仕様：
- 30秒間のまばたき回数を計測
- ニックネームとともに記録をCSVに保存
- リーダーボード表示

制限事項：
- カメラ権限が必要
- 1人の顔のみ対応
- pandas 1.4以降対応

@TODO: 複数人対応、まばたき検出の精度向上
@FIXME: カメラ認識失敗時の詳細なエラー処理
"""
import streamlit as st
import cv2
import mediapipe as mp
import time
import pandas as pd

# Constants for blink detection
eye_aspect_ratio_threshold = 0.2  # adjust as needed
eye_aspect_ratio_consecutive_frames = 2

# MediaPipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Indices for eye landmarks (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Utility to compute EAR
import numpy as np
def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    pts = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_indices]
    # horizontal
    A = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    # vertical
    B = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    C = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    ear = (B + C) / (2.0 * A)
    return ear

# File to store leaderboard
data_file = 'leaderboard.csv'

# Initialize leaderboard storage if not exists
def init_leaderboard():
    try:
        pd.read_csv(data_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['nickname', 'blinks', 'timestamp'])
        df.to_csv(data_file, index=False)

init_leaderboard()

# Streamlit UI
st.title("まばたき回数チャレンジ")
page = st.sidebar.selectbox("ページを選択", ["ゲーム", "リーダーボード"])

if page == "ゲーム":
    st.header("まばたきチャレンジ")
    # ニックネーム登録
    nickname_input = st.text_input("ニックネームを入力してください", "")
    if 'nickname' not in st.session_state:
        st.session_state['nickname'] = ""
    register_button = st.button(
        "ニックネームを登録",
        disabled=not nickname_input
    )
    if register_button and nickname_input:
        st.session_state['nickname'] = nickname_input
        st.rerun()
    # ニックネーム未登録時は以降非表示
    if not st.session_state['nickname']:
        st.info("ニックネームを登録してください。")
        st.stop()
    # セッション状態管理
    if 'timer_running' not in st.session_state:
        st.session_state['timer_running'] = False
    if 'start_time' not in st.session_state:
        st.session_state['start_time'] = None
    if 'blink_count' not in st.session_state:
        st.session_state['blink_count'] = 0
    if 'consec_frames' not in st.session_state:
        st.session_state['consec_frames'] = 0
    if 'reset_flag' not in st.session_state:
        st.session_state['reset_flag'] = False
    # カメラ映像表示
    st_frame = st.empty()
    if not st.session_state['timer_running']:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            st_frame.image(frame, channels="BGR")
        cap.release()
        start_button = st.button(
            "スタート",
            disabled=False
        )
        if start_button:
            st.session_state['timer_running'] = True
            st.session_state['start_time'] = time.time()
            st.session_state['blink_count'] = 0
            st.session_state['consec_frames'] = 0
            st.session_state['reset_flag'] = False
            st_frame.empty()  # スタート直後に静止画を消す
            st.rerun()
        st.stop()
    # リセットボタン
    reset_button = st.button("リセット") if st.session_state['timer_running'] else None
    if reset_button:
        st.session_state['timer_running'] = False
        st.session_state['start_time'] = None
        st.session_state['blink_count'] = 0
        st.session_state['consec_frames'] = 0
        st.session_state['reset_flag'] = True
        st.rerun()
    # タイマー・計測処理
    if st.session_state['timer_running']:
        cap = cv2.VideoCapture(0)
        col1, col2 = st.columns(2)
        with col1:
            timer_display = st.empty()
        with col2:
            count_display = st.empty()
        st_frame = st.empty()
        while True:
            elapsed = time.time() - st.session_state['start_time']
            if elapsed >= 20:
                break
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            h, w, _ = frame.shape
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0
                if ear < eye_aspect_ratio_threshold:
                    st.session_state['consec_frames'] += 1
                else:
                    if (
                        st.session_state['consec_frames'] >= eye_aspect_ratio_consecutive_frames
                    ):
                        st.session_state['blink_count'] += 1
                    st.session_state['consec_frames'] = 0
            timer_display.metric(
                "残り時間 (秒)",
                f"{int(20 - elapsed)}"
            )
            count_display.metric(
                "まばたき回数",
                st.session_state['blink_count']
            )
            st_frame.image(
                frame,
                channels="BGR",
                use_column_width=True
            )
            if st.session_state['reset_flag']:
                break
        cap.release()
        if not st.session_state['reset_flag']:
            # 保存
            df = pd.read_csv(data_file)
            new_row = pd.DataFrame([
                {
                    'nickname': st.session_state['nickname'],
                    'blinks': st.session_state['blink_count'],
                    'timestamp': pd.Timestamp.now()
                }
            ])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(data_file, index=False)
            # 順位計算
            df_sorted = df.sort_values(by='blinks', ascending=False)
            df_sorted['順位'] = df_sorted['blinks'].rank(method='min', ascending=False).astype(int)
            my_row = df_sorted[df_sorted['nickname'] == st.session_state['nickname']].tail(1)
            my_rank = int(my_row['順位'].values[0]) if not my_row.empty else '-'
            # おめでとうメッセージ
            st.markdown(
                f"""
                <div style='background-color:#fffbe7;padding:2em 1em;border-radius:16px;border:2px solid #ffe066;'>
                    <h2 style='color:#ff9800;text-align:center;font-size:2.2em;font-weight:bold;margin-bottom:0.5em;'>🎉 おめでとう！！ {st.session_state['nickname']} さん 🎉</h2>
                    <p style='color:#333;text-align:center;font-size:1.5em;font-weight:bold;'>回数は <span style='color:#ff5722;font-size:2em;'>{st.session_state['blink_count']}</span> 回で <span style='color:#2196f3;font-size:2em;'>{my_rank} 位</span> だよ！！！</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 もう一度チャレンジ"):
                    st.session_state['timer_running'] = False
                    st.session_state['start_time'] = None
                    st.session_state['blink_count'] = 0
                    st.session_state['consec_frames'] = 0
                    st.session_state['reset_flag'] = False
                    st.rerun()
            with col2:
                if st.button("🏆 リーダーボードを見る"):
                    st.session_state['page'] = "リーダーボード"
                    st.rerun()
        st.session_state['timer_running'] = False
        st.session_state['start_time'] = None
        st.session_state['blink_count'] = 0
        st.session_state['consec_frames'] = 0
        st.session_state['reset_flag'] = False
        st.rerun()

elif page == "リーダーボード":
    st.header("リーダーボード")
    df = pd.read_csv(data_file)
    if df.empty:
        st.info("まだプレイされた記録がありません。")
    else:
        # 回数降順でソート
        df_sorted = df.sort_values(by='blinks', ascending=False)
        # ランキング列（同順位対応）
        df_sorted['順位'] = df_sorted['blinks'].rank(
            method='min', ascending=False
        ).astype(int)
        # 日本語表記の列名で表示
        df_display = df_sorted[['順位', 'nickname', 'blinks']]
        df_display = df_display.rename(
            columns={'nickname': 'ニックネーム', 'blinks': '回数'}
        )
        st.table(
            df_display.set_index('順位')
        )
