# %% [markdown]
# LINEトークを解析してみよう！
# Pythonian 20-08-29

# %%
# ======================================
#      ライブラリ
# ======================================
import codecs
import re
import datetime
from time import strftime, gmtime
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import copy
import os
# 自然言語処理
from collections import Counter
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
import emoji
import string
# おまけ
import wordcloud
# %%
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Meirio', 'Hiragino Maru Gothic Pro', 'Yu Gothic',
                               'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']
plt.rcParams["font.size"] = 16

# %%
# ======================================
#      定数
# ======================================
FILTER_FILE = "unwanted_word.txt"
FONT_PATH = "C:\\Windows\\Fonts\\meiryo.ttc"  # for Word Cloud


# %%
# ======================================
#      クラス
# ======================================
class LineTalkAnalyzer:
    """Lineのトークを日付やメッセージ、発言者に分解するクラス"""
    (COL_DATETIME, COL_DAY, COL_WHO, COL_MSG) = (0, 1, 2, 3)
    MEDIA = ["PC", "Phone"]  # PCとスマートフォンでエクスポートされる形式が若干異なる
    FILE_STATISTICS = "statistics.txt"  # 統計情報を保存するファイル名
    FILE_INCL_CHARS = "incl_chars.png"  # 発言量のプロットを保存するファイル名
    FILE_INTARVAL = "interval.png"      # 返信にかかる時間のプロットを保存するファイル名

    def __init__(self, media="PC", save_dir="result/"):
        if media not in self.MEDIA:
            print("media error: {0}".format(media))
            return
        self.media = media  # PC と スマホ(Phone)で形式が異なる
        self.readout = True  # テキストの最初の不要なデータを無視
        self.save_dir = save_dir  # データの保存ディレクトリ
        os.makedirs(save_dir, exist_ok=True)

        self.talk = []  # date,day,time,who,msg
        self.call = []  # 電話の履歴
        self.stamp = []  # スタンプの送信
        self.image = []  # 画像の送信
        self.whos = []  # トークメンバー
        if media == self.MEDIA[0]:  # PC
            self.pat_date = re.compile(r"^(20\d\d.\d\d.\d\d) (.+)曜日")
            self.pat_talk_begin = re.compile(r"^(\d\d:\d\d) (.+?) (.+)")
            self.strpfmt = "%Y.%m.%d %H:%M"
            self.strpfmt_date = "%Y.%m.%d"
            self.pat_time = re.compile(r"^(\d\d?):(\d{2}):?(\d?\d?)$")
        else:  # Phone
            self.pat_date = re.compile(r"^(20\d\d\/\d?\d\/\d?\d)\((.)\)")
            self.pat_talk_begin = re.compile(r"^(\d?\d:\d\d)	(.+?)	(.+)")
            self.strpfmt = "%Y/%m/%d %H:%M"
            self.strpfmt_date = "%Y/%m/%d"
            self.pat_time = re.compile(r"^通話時間(\d\d?):(\d{2}):?(\d?\d?)$")
        # 読み飛ばすパターン
        self.pat_url = re.compile(
            r"(https?:\/\/[=\w!?\/+\-_~;.,*&@#$%()'[\]]+)")  # URLマッチ
        self.pat_add = re.compile(r".+がグループに参加しました。")
        self.pat_invite = re.compile(r".+を招待しました。")
        return

    def parse_raw_talk(self, talk):
        """文字列データ化したトークを日付、曜日、時間、発言者、メッセージに分解"""
        date = day = time = who = msg = ""
        for line in talk:
            # 改行削除
            line = line.replace("\r\n", "")
            line = line.replace("\n", "")
            if line == "":
                continue

            # 1. 日付マッチ
            # --------------
            m = self.pat_date.match(line)
            if m is not None:
                # 初めて日付を得たら読み飛ばしを終了
                if self.readout:
                    self.readout = False
                date, day = m.groups()[:2]
                continue
            elif self.readout:
                continue

            # 2. 発言者が変わった時のトーク開始を取得
            # ----------------------------------------
            m = self.pat_talk_begin.match(line)
            if m is not None:
                time, who, msg = m.groups()[:3]
                msg = self._sanitize_msg(msg)
                # TODO: たまに変なエラーが出る...
                try:
                    dt = datetime.datetime.strptime(
                        "{0} {1}".format(date, time), self.strpfmt)
                    self._put(dt, day, who, msg)
                except ValueError:
                    print("value Error: ", date, time)
                continue

            # 3. 発言者が変わらず、トーク継続を取得
            # ----------------------------------------
            msg = self._sanitize_msg(line)
            # TODO: たまに変なエラーが出る...
            try:
                dt = datetime.datetime.strptime(
                    "{0} {1}".format(date, time), self.strpfmt)
                self._put(dt, day, who, msg)
            except:
                print("ERROR: ", end="")
                print(date, day, time, msg)

        self._separate_phone_call()
        self._separate_stamp()
        self._separate_image()
        self._unify_date()

        self.whos = set([x[self.COL_WHO] for x in self.talk])

    def print(self):
        """分解したトークを表示"""
        for line in self.talk:
            print("{0}({1}) {2}: {3}".format(
                line[0].strftime("%m/%d %H:%M"), *line[1:]))

    # ==================================================
    #   情報表示関数
    # ==================================================
    def show_statistics(self):
        """解析トークの統計情報を表示"""
        num_lines = {}
        num_chars = {}
        # 人ごとのメッセージ数
        for who in self.whos:
            num_lines[who] = len(self.get_msg(who=who, only_msg=False))
            num_chars[who] = len(self.get_msg(who=who, only_msg=True))

        msg = "=== 統計情報 ===\n"
        msg += "メンバー: "
        for who in self.whos:
            msg += "{0} ".format(who)
        msg += "\n期間: {0}～{1}".format(self.talk[0][self.COL_DATETIME].strftime(self.strpfmt_date),
                                      self.talk[-1][self.COL_DATETIME].strftime(self.strpfmt_date))
        msg += "\n会話統計"
        for who in self.whos:
            msg += "\n {0}: {1}行 {2}文字".format(
                who, num_lines[who], num_chars[who])
        msg += "\n電話時間: {0}".format(self._call_time())
        msg += "\nスタンプ: {0}回".format(len(self.stamp))
        msg += "\n画像送信: {0}回".format(len(self.image))
        print(msg)
        with open(self.save_dir + self.FILE_STATISTICS, "w") as wf:
            wf.write(msg)

    def show_incl_chars(self):
        """発言量（文字数）をプロット"""
        new_talk = []
        each_talk = {}

        for line in self.talk:
            x = line[self.COL_DATETIME]
            y = len(line[self.COL_MSG])
            new_talk.append([line[self.COL_WHO], x, y])

        xfmt = mdates.DateFormatter("%y/%m/%d")
        plt.figure(figsize=(10, 8))

        # 日ごと と 累計
        for who in self.whos:
            each_talk[who] = [x[1:] for x in new_talk if x[0] == who]
            x = [x[0] for x in each_talk[who]]
            y = [x[1] for x in each_talk[who]]
            yy = [sum(y[:x + 1]) for x in range(len(y))]
            ax = plt.subplot(2, 1, 1)
            ax.plot(x, y)
            ax = plt.subplot(2, 1, 2)
            ax.plot(x, yy)
        ax.legend(self.whos)

        ax_title = ["個別", "累計"]
        for i in range(2):
            ax = plt.subplot(2, 1, i+1)
            ax.xaxis.set_major_formatter(xfmt)
            ax.set_xlim(new_talk[0][1], new_talk[-1][1])
            ax.grid()
            ax.set_ylabel("# of characters")
            ax.set_title(ax_title[i])
        plt.savefig(self.save_dir + self.FILE_INCL_CHARS)
        plt.show(block=False)

    def show_interval(self, each=False):
        """どのくらいのスパンで返信しているかをプロット"""
        dt = self.talk[0][self.COL_DATETIME]
        who = ""
        diffs = []

        # 人が変わったらdiffを取る
        for line in self.talk:
            new_who = line[self.COL_WHO]
            new_dt = line[self.COL_DATETIME]
            if new_who != who:
                diff = new_dt - dt
                diffs.append([who, (diff.seconds) / 60 / 60])  # hour
                who = new_who
            dt = new_dt

        diffs = diffs[1:]  # 最初はゴミデータが入る(0番目には何も入っていないので差分が大きい)

        plt.figure(figsize=(8, 6))
        if each:
            for who in self.whos:
                data = [x[1] for x in diffs if x[0] == who]
                plt.hist(data, bins=64, alpha=0.5)
            plt.legend(self.whos)
        else:
            plt.hist([x[1] for x in diffs], bins=64)
        plt.xticks(list(range(0, 300 + 1, 6)))

        # 6時間単位で表示
        x_max = max([x[1] for x in diffs])
        x_max = math.ceil(x_max / 6) * 6
        plt.xlim([0, x_max])
        plt.xlabel("Elapsed time [h]")
        plt.ylabel("# of occurence")
        plt.title("Talk intarval")
        plt.grid()
        plt.savefig(self.save_dir + self.FILE_INTARVAL)
        plt.show(block=False)

    # ==================================================
    #   getter
    # ==================================================
    def get_msg(self, who=None, only_msg=False):
        if who is not None:
            talk = [x for x in self.talk if x[self.COL_WHO] == who]
        else:
            talk = self.talk

        if only_msg:
            msgs = [x[self.COL_MSG] for x in talk]
            msgs = "".join(msgs).replace(" ", "")
            return msgs
        else:
            return talk

    def get_members(self):
        return list(self.whos)

    # ==================================================
    #   補助関数
    # ==================================================
    def _put(self, dt, day, who, msg):
        if msg != "":
            self.talk.append([dt, day, who, msg])

    def _separate_phone_call(self):
        """メッセージから電話履歴を分離して保存"""
        self.call = [
            x for x in self.talk if self._is_phone_call(x[self.COL_MSG])]
        for call in self.call:
            self.talk.remove(call)

    def _separate_stamp(self):
        """メッセージからスタンプを分離して保存"""
        if self.media == self.MEDIA[0]:
            self.stamp = [x for x in self.talk if x[self.COL_MSG] == "スタンプ"]
            self.talk = [x for x in self.talk if x[self.COL_MSG] != "スタンプ"]
        else:
            self.stamp = [x for x in self.talk if x[self.COL_MSG] == "[スタンプ]"]
            self.talk = [x for x in self.talk if x[self.COL_MSG] != "[スタンプ]"]

    def _separate_image(self):
        """メッセージから画像送信を分離して保存"""
        if self.media == self.MEDIA[0]:
            self.image = [x for x in self.talk if x[self.COL_MSG] == "画像"]
            self.talk = [x for x in self.talk if x[self.COL_MSG] != "画像"]
        else:
            self.image = [x for x in self.talk if x[self.COL_MSG] == "[写真]"]
            self.talk = [x for x in self.talk if x[self.COL_MSG] != "[写真]"]

    def _sanitize_msg(self, msg):
        """メッセージから不要な文字列を除外"""
        # URLを除去
        if self.pat_add.search(msg) is not None:
            return ""
        if self.pat_invite.search(msg) is not None:
            return ""
        new = self.pat_url.sub("", msg)
        new = new.replace("　", "")
        new = new.replace(" ", "")
        new = new.replace("新しいノートを作成しました。", "")
        new = new.replace("不在着信", "")
        new = new.replace("キャンセル", "")
        return new

    def _is_phone_call(self, msg):
        return (self.pat_time.match(msg) is not None)

    def _call_time(self):
        """電話時間の合計を計算"""
        def str2sec(str_time):
            time = 0
            coef = [1, 60, 3600]

            m = self.pat_time.match(str_time)
            sep_time = m.groups()
            sep_time = [x for x in sep_time if x != ""]
            for idx, dat in enumerate(sep_time[:: -1]):
                if dat != "":
                    time += int(dat) * coef[idx]
            return time

        time = sum(list(map(lambda x: str2sec(x[self.COL_MSG]), self.call)))
        time = strftime("%H:%M:%S", gmtime(time))
        return time

    def _unify_date(self):
        """ばらばらの日付をまとめる"""
        dt = day = who = msg = ""
        talk = []
        for line in self.talk:
            new_dt = line[self.COL_DATETIME]
            new_who = line[self.COL_WHO]
            if dt == new_dt and who == new_who:
                msg += line[self.COL_MSG]
            else:
                talk.append([dt, day, who, msg])
                dt = new_dt
                who = new_who
                msg = line[self.COL_MSG]
                day = line[self.COL_DAY]
        else:
            talk.append([dt, day, who, msg])
            talk.remove(["", "", "", ""])  # 1行目には必ずゴミデータが入る
        self.talk = talk


# %%
class NumericReplaceFilter(TokenFilter):
    """名詞中の数(漢数字を含む)を全て0に置き換えるTokenFilterの実装"""

    def apply(self, tokens):
        for token in tokens:
            parts = token.part_of_speech.split(',')
            if (parts[0] == '名詞' and parts[1] == '数'):
                token.surface = '0'
                token.base_form = '0'
                token.reading = 'ゼロ'
                token.phonetic = 'ゼロ'
            yield token


class NaturalLanguageProcessing:
    """janomeによる自然言語処理により、絵文字や名詞の頻度を分析する"""
    # MEMO: %?!~ を残す (これは好みで)
    pat_simb = re.compile(r"[\"-$]|[&-\/]|[:->]|@|[[-`]|[{-~]|（|）")
    wc_max_words = 130  # ワードクラウドに表示する文字の最大数
    # 絵文字をmaptlotlibで処理するためのフォント
    # Win10標準の"SEGOE UI EMOJI"だけだと表示のされ方がイマイチな絵文字があるため２つ
    FONT1 = "c:\\windows\\fonts\\seguiemj.ttf"
    fp1 = fm.FontProperties(fname=FONT1)
    FONT2 = "C:\\Users\\User\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Symbola_hint.ttf"
    if os.path.exists(FONT2):
        fp2 = fm.FontProperties(fname=FONT2)
    else:
        fp2 = None

    def __init__(self, save_dir="result/"):
        self.save_dir = save_dir
        # 解析器
        token_filters = [POSKeepFilter(['名詞']),
                         NumericReplaceFilter(),
                         TokenCountFilter()]
        char_filters = [UnicodeNormalizeCharFilter()]
        self.ana = Analyzer(char_filters=char_filters,
                            token_filters=token_filters)
        # 個人ごとの解析結果を入れる
        self.noun = {}
        self.emoji = {}

        # サニタイズフィルタ
        self.filter = []
        with codecs.open(FILTER_FILE, "r", "utf-8") as rf:
            for line in rf.readlines():
                self.filter.append(line.rstrip())

        # Word Cloud
        self.wc = wordcloud.WordCloud(
            background_color="white",
            max_words=self.wc_max_words,
            max_font_size=550,
            width=1920,
            height=1080,
            # mask=mask,
            font_path=FONT_PATH)

    def analyze(self, msg, who="all"):
        """whoで個別のデータを貯める。マージは別関数で"""
        # 絵文字の分離
        emo = [x for x in msg if x in emoji.UNICODE_EMOJI]
        msg = "".join([x for x in msg if x not in emoji.UNICODE_EMOJI])

        # 前処理 (不要な文字の除去)
        msg = self.pat_simb.sub("", msg)

        self.emoji[who] = Counter(emo)
        noun = list(self.ana.analyze(msg))
        noun = self._sanitize_noun(noun)
        noun.sort(key=lambda x: - x[1])
        self.noun[who] = noun

    def show(self, who, min_freq=0):
        for key, val in self.noun[who]:
            if val >= min_freq:
                print("{0}: {1}".format(key, val))
                # print("{0}".format(key))

    def merge_dict(self):
        """個人ごとの辞書をマージして"all"の辞書を作る"""
        n = Counter()
        e = Counter()
        for who in self.noun.keys():
            n += Counter(dict(self.noun[who]))
            e += Counter(dict(self.emoji[who]))
        self.noun["all"] = list(dict(n).items())
        self.emoji["all"] = list(dict(e).items())

    def get_noun(self):
        return self.noun

    def produce_word_cloud(self, who, include_emoji=False):
        """ワードクラウドを生成する
        MEMO: 絵文字は使えない"""
        self.wc.generate_from_frequencies(dict(self.noun[who]))

        plt.figure(figsize=(10, 8))
        plt.imshow(self.wc, interpolation="bilinear")
        plt.axis("off")
        plt.show(block=False)
        self.wc.to_file(self.save_dir + "wc_{0}.png".format(who))

    def show_emoji_freq(self, min_freq=5):
        """絵文字の使用頻度を解析する;
        全員合わせてmin_freq回よりも多く使われた絵文字のみ抽出"""
        # keyのみを含む値0の辞書を作る
        all_dict = self.emoji["all"]
        all_dict = [x for x in all_dict if x[1] > min_freq]
        all_dict.sort(key=lambda x: - x[1])
        all_dict = dict(all_dict)
        empty_dict = dict.fromkeys(all_dict, 0)

        # 個人ごとの頻度を辞書にする
        keys = [x for x in self.emoji.keys() if x != "all"]
        each_dict = np.array([])
        for i, name in enumerate(keys):
            dic = dict(self.emoji[name].items())
            containter = copy.deepcopy(empty_dict)
            for key in dic.keys():
                if key in containter.keys():
                    containter[key] = dic[key]
            each_dict = np.append(each_dict, list(containter.values()))
        each_dict = each_dict.reshape(len(keys), -1)

        # 積算してグラフ化
        fig, ax = plt.subplots(figsize=(10, 8))
        xdata = range(each_dict.shape[1])
        for i in range(each_dict.shape[0]):
            bar = ax.bar(xdata, each_dict[i, :],
                         bottom=each_dict[:i, :].sum(axis=0))

        ax.set_ylim([0, max(all_dict.values()) + 15])

        height = list(all_dict.values())
        # 絵文字は表示が微妙なので2フォント使う
        for i, (rect, emo) in enumerate(zip(bar, all_dict.keys())):
            plt.annotate(emo,
                         (rect.get_x() + rect.get_width() / 2, height[i] + 2),
                         ha="center", va="bottom",
                         fontproperties=self.fp1, fontsize=20)
            if self.fp2 is not None:
                plt.annotate(emo,
                             (rect.get_x() + rect.get_width() /
                              2, height[i] + 7),
                             ha="center", va="bottom",
                             fontproperties=self.fp2)

        ax.set_xticks(xdata)
        ax.set_xticklabels(all_dict.values())

        ax.set(xlabel='絵文字', ylabel='回数')
        ax.legend(keys)
        ax.grid()
        fig.savefig(self.save_dir + "emoji_freq.png")
        plt.show(block=False)

    def _sanitize_noun(self, noun):
        """好ましくない結果を除外する"""
        # TODO: 統合すべき言葉を別途ファイルで管理したい
        noun = [x for x in noun if x[0] not in self.filter]

        # 同一視する言葉を統合する
        # "..." "......" がどちらも出てきたら統合する
        noun = dict(noun)
        if "......" in noun and "..." in noun:
            cnt = noun["......"] + noun["..."]
            noun.pop("......")
            noun["..."] = cnt

        # 分割された言葉を統合する（人名など）
        if "サイド" in noun and "チャネル" in noun:
            cnt = int((noun.pop("サイド") + noun.pop("チャネル"))/2)
            noun["サイドチャネル"] = cnt

        return list(noun.items())


# %%
def file2process(fname, media="PC"):
    with codecs.open(fname, "r", "utf-8") as rf:
        talk = rf.readlines()

    lta = LineTalkAnalyzer(media=media)
    nlp = NaturalLanguageProcessing()

    lta.parse_raw_talk(talk)
    lta.show_statistics()
    lta.show_incl_chars()
    lta.show_interval(each=True)

    for who in lta.get_members():
        msg = lta.get_msg(who=who, only_msg=True)
        nlp.analyze(msg, who=who)
        nlp.produce_word_cloud(who)
    nlp.merge_dict()
    nlp.produce_word_cloud("all")
    nlp.show_emoji_freq(min_freq=1)

    return lta, nlp


# %%
if __name__ == "__main__":
    fname = "[LINE] りんなとのトーク"
    lta, nlp = file2process(fname, media="Phone")
    # lta, nlp = file2process(fname, media="Phone")
    input()


# %% [markdown]
# https://qiita.com/s_fukuzawa/items/6f9c1a3d4c4f98ae6eb1
# https://qiita.com/yniji/items/2f0fbe0a52e3e067c23c
# https://ohke.hateblo.jp/entry/2017/11/02/230000
# https://qiita.com/turmericN/items/04cd0b40f91076f0ef42
# https://pira-nino.hatenablog.com/entry/2018/07/27/B%27z%E3%81%AE%E6%AD%8C%E8%A9%9E%E3%82%92Python%E3%81%A8%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%A7%E5%88%86%E6%9E%90%E3%81%97%E3%81%A6%E3%81%BF%E3%81%9F_%E3%80%9C%E3%83%87%E3%83%BC%E3%82%BF

# %%
