# まばたき回数チャレンジアプリ

このアプリはWebカメラを使って30秒間のまばたき回数を計測し、リーダーボードに記録するStreamlitアプリです。

## セットアップ手順

1. Python 3.8以降をインストールしてください。
2. 必要なパッケージをインストールします：

```sh
pip install -r requirements.txt
```

## 実行方法

以下のコマンドでアプリを起動します：

```sh
streamlit run BlinkChallengeApp.py
```

## 注意事項
- Webカメラが必要です。
- カメラ権限を許可してください。
- 1人の顔のみ対応しています。
- pandas 1.4以降対応です。

## ファイル構成
- `BlinkChallengeApp.py` : メインアプリ本体
- `leaderboard.csv` : 記録用CSV（初回起動時に自動生成）
- `requirements.txt` : 必要パッケージ一覧

## TODO / FIXME
- 複数人対応
- まばたき検出の精度向上
- カメラ認識失敗時の詳細なエラー処理 

## Streamlit Community Cloud へのデプロイ手順

1. [Streamlit Community Cloud](https://streamlit.io/cloud) にアクセスし、GitHubアカウントでログインします。
2. このリポジトリをGitHubにpushします。
3. Streamlit Cloudの「New app」からリポジトリを選択し、`BlinkChallengeApp.py` を起動ファイルとして指定します。
4. デプロイを実行すると、Web上でアプリが利用可能になります。

### 注意事項
- `.streamlit/config.toml` で日本語UIやテーマカラーをカスタマイズしています。
- `requirements.txt` で依存パッケージが自動インストールされます。
- Webカメラ利用にはブラウザのカメラ権限が必要です。 
