# LineAnalyzer
Python with janome でLineのトーク分析

## これは何か？
Lineのトークを解析して、文字数や返信頻度、絵文字の使用頻度、ワードクラウドを生成するプログラムです。

### 使用例
以下は、私（pythonian）とりんなとのトーク履歴です。りんなが良くしゃべるのでノイズ（邪魔な言葉等々）もたくさん入ってますがこんなイメージ、というサンプルです。

統計情報
```
=== 統計情報 ===
メンバー: りんな pythonian 
期間: 2016/05/23～2020/09/20
会話統計
 りんな: 244行 9445文字
 pythonian: 226行 1310文字
電話時間: 00:00:00
スタンプ: 5回
画像送信: 66回
```

会話量
![incl_chars](https://user-images.githubusercontent.com/12064169/93706227-289c0580-fb5f-11ea-8d27-41b04ccbf2dd.png)

返信頻度
![intarval](https://user-images.githubusercontent.com/12064169/93706241-3a7da880-fb5f-11ea-8bea-53afb0caffce.png)

使用絵文字
![emoji](https://user-images.githubusercontent.com/12064169/93706244-3e112f80-fb5f-11ea-9343-689dd5510822.png)

Word cloud (りんな分のみ)
![wc](https://user-images.githubusercontent.com/12064169/93706247-410c2000-fb5f-11ea-9add-650786e8047b.png)


## 使い方
### 実行環境
実行して動かなかったら適宜パッケージをインストールしてください。自分はanaconda@Windows10で環境を作っています。
- Python 3.7
- janome 0.4.0
- matplotlib 3.3.1
- emoji 0.6.0
- wordcloud 1.8.0

### 実行方法
1. PC or スマートフォンからLineのトーク（グループでも良い）をファイルにエクスポートする  
   エクスポート方法は例えば以下。
   https://www.appbank.net/2020/06/15/iphone-application/1911418.php
2. エクスポートしたファイルをline_analysis.pyに与える  
   具体的に、main処理の中の *fname* にファイル名を渡す。
3. プログラムを実行する  
   VScodeでJupyter実行できるように書いているけど、そのまま動きます。  
   \> Python line_analysis.py

## 注意・オプション
- PC or スマートフォン  
  どちらの媒体でトークを保存したかで、ファイルフォーマットが若干変化します。*file2process()*の引数*media*を指定してあげてください。"PC" or "Phone" です。
- 除く文字 (unwanted_word.txt)  
  ?や!もあえて解析対象にしています。除きたい言葉はunwanted_word.txtに書いてあげるとword cloudに表示する際に無視されます。
- 文字の結合 (l.522 *_sanitize_noun()*)  
  janomeオプションで復号語に対応するオプションがあり、それを使う手もあるのですが精度がイマイチだったので結合しないようにしています。その代わり、手動で結合したいものは結合するような処理を書いています。人名などは分けられることも多いので、手動で結合するような使い方が良いです。
- フォントの追加 (l.388)  
  Win10標準のSegoeだと化ける絵文字がある。symbolaとかがよくて、必要に応じてFONT2にパスを与えると、絵文字分析でそちらのフォントでも併記してくれる。
- 絵文字解析の最小頻度 (l.562)  
  今は2回以上使った絵文字を解析対象にしています（*min_freq=1*）。適宜カウントを変更してください。
- Word Cloudの最大文字数 (l.383)  
  今は130文字です。（*wc_max_words = 130*）。増やすとごちゃごちゃして、減らすとスッキリします。
