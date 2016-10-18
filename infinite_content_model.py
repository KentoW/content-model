# -*- coding: utf-8 -*-
# 無限コンテンツモデル(infinite content model)
import sys
import math
import random
import argparse
import scipy.special
from collections import defaultdict

BOS = 0
EOS = -1
UNLABEL = -2

class ICM:
    def __init__(self, data):
        self.corpus_file = data
        self.target_word = defaultdict(int)
        self.corpus = []
        doc = {"comment":"", "blocks":[]}
        for strm in open(data, "r"):
            if strm.startswith("#"):
                if len(doc["blocks"]) > 0:
                    doc["comment"] = comment
                    self.corpus.append(doc)
                    doc = {"comment":"", "blocks":[]}
                comment = strm.strip()
            else:
                block = strm.strip().split(" ")
                doc["blocks"].append(block[::])
                for v in block:
                    self.target_word[v] += 1
        if len(doc["blocks"]) > 0:
            doc["comment"] = comment
            self.corpus.append(doc)
            doc = {"comment":"", "blocks":[]}
        self.V = float(len(self.target_word))
        # 潜在変数値
        self.topic = defaultdict(lambda: defaultdict(int))                          # m番目の文書のb番目のブロックのトピック
        # 遷移分布
        self.trans_freq = defaultdict(lambda: defaultdict(float))
        self.trans_sum = defaultdict(float)
        self.trans_to = defaultdict(float)
        # 単語分布
        self.word_freq = defaultdict(lambda: defaultdict(float))
        self.word_sum = defaultdict(float)

    def set_param(self, alpha, beta, N, converge):
        self.alpha = alpha
        self.beta = beta
        self.K = 0
        self.N = N
        self.converge = converge

    def initialize(self):
        for m, doc in enumerate(self.corpus):
            for b, block in enumerate(doc["blocks"]):
                self.topic[m][b] = UNLABEL

    def likelihood(self):
        likelihoods = []
        for doc in self.corpus:
            likelihood = 0.0
            score = defaultdict(lambda: defaultdict(float))
            for b, block in enumerate(doc["blocks"]):
                if b == 0:
                    for z in xrange(1, self.K+1):
                        L_trans = math.log((self.trans_freq[BOS][z] + self.alpha) / (self.trans_sum[BOS] + self.alpha*self.K))
                        L_block = 0.0
                        for v in block:
                            L_block += math.log((self.word_freq[z][v] + self.beta) / (self.word_sum[z] + self.beta*self.V))
                        score[b][z] = L_trans + L_block
                else:
                    for z in xrange(1, self.K+1):
                        prev_score_sum = 0.0
                        max_log = -999999.9
                        for prev_z in xrange(1, self.K+1):
                            if max_log < score[b-1][prev_z]:
                                max_log = score[b-1][prev_z]
                        for prev_z in xrange(1, self.K+1):
                            L_trans = math.log((self.trans_freq[prev_z][z] + self.alpha) / (self.trans_sum[prev_z] + self.alpha*self.K))
                            prev_score_sum += math.exp(score[b-1][prev_z] + L_trans - max_log)
                        L_block = 0.0
                        for v in block:
                            L_block += math.log((self.word_freq[z][v] + self.beta) / (self.word_sum[z] + self.beta*self.V))
                        score[b][z] = math.log(prev_score_sum) + L_block + max_log
            max_log = -999999.9
            prev_score_sum = 0.0
            for prev_z in xrange(1, self.K+1):
                if max_log < score[b][prev_z]:
                    max_log = score[b][prev_z]
            for prev_z in xrange(1, self.K+1):
                prev_score_sum += math.exp(score[b][prev_z] - max_log)
            likelihood = math.log(prev_score_sum) + max_log
            likelihoods.append(likelihood)
        return sum(likelihoods)/len(likelihoods)

    def learn(self):
        self.initialize()
        self.lkhds = []
        for i in xrange(self.N):
            self.gibbs_sampling()
            sys.stderr.write("iteration=%d/%d K=%s alpha=%s beta=%s\n"%(i+1, self.N, self.K, self.alpha, self.beta))
            if i % 10 == 0:
                self.n = i+1
                self.lkhds.append(self.likelihood())
                sys.stderr.write("%s : likelihood=%f\n"%(i+1, self.lkhds[-1]))
                if len(self.lkhds) > 1:
                    diff = self.lkhds[-1] - self.lkhds[-2]
                    if math.fabs(diff) < self.converge:
                        break
        self.n = i+1

    def gibbs_sampling(self):
        for m, doc in enumerate(self.corpus):
            for b, block in enumerate(doc["blocks"]):
                self.sample_topic(m, b) # コーパス中のm番目の文書のb番目のブロックのトピックをサンプリング
        nominator = 0.0     # ハイパーパラメータ αの更新
        denominator = 0.0
        for prev_z in xrange(0, self.K+1):
            for z in xrange(1, self.K+1):
                nominator += scipy.special.digamma(self.trans_freq[prev_z][z] + self.alpha)
            denominator += scipy.special.digamma(self.trans_sum[prev_z] + self.alpha*self.K)
        nominator -= (self.K+1)*self.K*scipy.special.digamma(self.alpha)
        denominator = self.K*denominator - (self.K+1)*self.K*scipy.special.digamma(self.alpha*self.K)
        self.alpha = self.alpha * (nominator / denominator)
        nominator = 0.0         # ハイパーパラメータ βの更新
        denominator = 0.0
        for z in xrange(1, self.K+1):
            for v in self.target_word:
                nominator += scipy.special.digamma(self.word_freq[z][v] + self.beta)
            denominator += scipy.special.digamma(self.word_sum[z] + self.beta*self.V)
        nominator -= (self.K*self.V*scipy.special.digamma(self.beta))
        denominator = (self.V*denominator) - (self.K*self.V*scipy.special.digamma(self.beta*self.V))
        self.beta = self.beta * (nominator / denominator)

    def sample_topic(self, m, b):
        z = self.topic[m][b]                        # Step1: カウントを減らす
        prev_z = self.topic[m].get(b-1, BOS)
        next_z = self.topic[m].get(b+1, EOS)
        if z != UNLABEL:
            self.trans_freq[prev_z][z] -= 1.0
            self.trans_sum[prev_z] -= 1.0
            self.trans_to[z] -= 1.0
            if next_z != EOS:
                self.trans_freq[z][next_z] -= 1.0
                self.trans_sum[z] -= 1.0
                self.trans_to[next_z] -= 1
            for v in self.corpus[m]["blocks"][b]:
                self.word_freq[z][v] -= 1.0
                self.word_sum[z] -= 1.0
            if self.trans_to[z] == 0:
                self.fill_K(z)

        z = self.topic[m][b]                        # Step2.1: 既存の事後分布の計算
        prev_z = self.topic[m].get(b-1, BOS)
        next_z = self.topic[m].get(b+1, EOS)
        p_z = defaultdict(float)                    
        n_b_v = defaultdict(float)
        n_b = 0.0
        for v in self.corpus[m]["blocks"][b]:
            n_b_v[v] += 1.0
            n_b += 1.0
        for z in xrange(1, self.K+1):
            p_z[z] = math.log((self.trans_freq[prev_z][z] + self.alpha)/ (self.trans_sum[prev_z] + self.alpha*self.K))
            if next_z != EOS and next_z != UNLABEL:
                I1 = 0.0
                I2 = 0.0
                if (prev_z == z == next_z): I1 = 1.0
                if (prev_z == z): I2 = 1.0
                p_z[z] += math.log((self.trans_freq[z][next_z] + I1 + self.alpha)/ (self.trans_sum[z] + I2 + self.alpha*self.K))
            first = math.lgamma(self.word_sum[z] + self.beta*self.V) - math.lgamma(self.word_sum[z] + n_b + self.beta*self.V)
            second = 0.0
            for v in n_b_v.iterkeys():
                second += math.lgamma(self.word_freq[z][v] + n_b_v[v] + self.beta) - math.lgamma(self.word_freq[z][v] + self.beta)
            p_z[z] += (first + second)

        p_z[self.K+1] = math.log(self.alpha / (self.trans_sum[prev_z] + self.alpha))        # Step2.2: 新しい事後分布
        if next_z != -2:
            if next_z != -1:
                I2 = 0.0
                if (prev_z == self.K+1):
                    I2 = 1.0
                p_z[self.K+1] += math.log(self.alpha / (self.trans_sum[prev_z] + I2 + self.alpha))
        p_z[self.K+1] += (math.lgamma(self.beta*self.V) - math.lgamma(n_b + self.beta*self.V))
        for v in n_b_v.iterkeys():
            p_z[self.K+1] += (math.lgamma(n_b_v[v] + self.beta) - math.lgamma(self.beta))

        max_log = max(p_z.itervalues())     # オーバーフロー対策
        for z in p_z:
            p_z[z] = math.exp(p_z[z]-max_log)
        new_z = self.sample_one(p_z)                # Step3: サンプル
        if new_z == self.K+1:
            self.K = self.K+1

        self.topic[m][b] = new_z          # Step4: カウントを増やす
        self.trans_freq[prev_z][new_z] += 1.0
        self.trans_sum[prev_z] += 1.0
        self.trans_to[new_z] += 1.0
        if next_z != EOS and next_z != UNLABEL:
            self.trans_freq[new_z][next_z] += 1.0
            self.trans_sum[new_z] += 1.0
            self.trans_to[next_z] += 1.0
        for v in self.corpus[m]["blocks"][b]:
            self.word_freq[new_z][v] += 1.0
            self.word_sum[new_z] += 1.0

    def fill_K(self, fill_z):
        for m, doc in enumerate(self.corpus):
            for b, block in enumerate(doc["blocks"]):
                if self.topic[m][b] >= fill_z:
                    self.topic[m][b] = self.topic[m][b]-1
        for z in xrange(1, self.K+1):
            if z == self.K:
                del self.trans_to[z]
            elif z >= fill_z:
                self.trans_to[z] = self.trans_to[z+1]
        for prev_z in xrange(0, self.K+1):
            for z in xrange(1, self.K+1):
                if z == self.K:
                    del self.trans_freq[prev_z][z]
                elif z >= fill_z:
                    self.trans_freq[prev_z][z] = self.trans_freq[prev_z][z+1]
        for prev_z in xrange(0, self.K+1):
            if prev_z == self.K:
                del self.trans_freq[prev_z]
                del self.trans_sum[prev_z]
            elif prev_z >= fill_z:
                for z in xrange(1, self.K):
                    self.trans_freq[prev_z][z] = self.trans_freq[prev_z+1][z]
                self.trans_sum[prev_z] = self.trans_sum[prev_z+1]
        for z in xrange(1, self.K+1):
            if z == self.K:
                del self.word_sum[z]
                del self.word_freq[z]
            elif z >= fill_z:
                self.word_sum[z] = self.word_sum[z+1]
                self.word_freq[z] = defaultdict(float)
                for v, freq in self.word_freq[z+1].iteritems():
                    self.word_freq[z][v] = freq
        self.K = self.K - 1

    def sample_one(self, prob_dict):
        z = sum(prob_dict.values())                     # 確率の和を計算
        remaining = random.uniform(0, z)                # [0, z)の一様分布に従って乱数を生成
        for state, prob in prob_dict.iteritems():       # 可能な確率を全て考慮(状態数でイテレーション)
            remaining -= prob                           # 現在の仮説の確率を引く
            if remaining < 0.0:                         # ゼロより小さくなったなら，サンプルのIDを返す
                return state

    def output_model(self):
        print "model\thidden_content_model"
        print "@parameter"
        print "corpus_file\t%s"%self.corpus_file
        print "hyper_parameter_alpha\t%f"%self.alpha
        print "hyper_parameter_beta\t%f"%self.beta
        print "number_of_hidden_variable\t%d"%self.K
        print "number_of_iteration\t%d"%self.n
        print "@likelihood"
        print "initial likelihood\t%s"%(self.lkhds[0])
        print "last likelihood\t%s"%(self.lkhds[-1])
        print "@vocaburary"
        for v in self.target_word:
            print "target_word\t%s"%v
        print "@count"
        for prev_z, dist in self.trans_freq.iteritems():
            print 'trans_sum\t%s\t%d' % (prev_z, self.trans_sum[prev_z])
            for z, freq in dist.iteritems():
                print 'trans_freq\t%s\t%s\t%d' % (prev_z, z, freq)
        for z, dist in self.word_freq.iteritems():
            print 'word_sum\t%s\t%d' % (z, self.word_sum[z])
            for v, freq in sorted(dist.iteritems(), key=lambda x:x[1], reverse=True):
                if int(freq) != 0:
                    print 'word_freq\t%s\t%s\t%d' % (z, v, freq)
        print "@data"
        for m, doc in enumerate(self.corpus):
            print doc["comment"]
            for b, block in enumerate(doc["blocks"]):
                z = self.topic[m][b]
                print "topic\t%s\t%s"%(z, " ".join(block))


def main(args):
    icm = ICM(args.data)
    icm.set_param(args.alpha, args.beta, args.N, args.converge)
    icm.learn()
    icm.output_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", dest="alpha", default=0.1, type=float, help="hyper parameter alpha")
    parser.add_argument("-b", "--beta", dest="beta", default=0.005, type=float, help="hyper parameter beta")
    parser.add_argument("-n", "--N", dest="N", default=1000, type=int, help="max iteration")
    parser.add_argument("-c", "--converge", dest="converge", default=0.01, type=str, help="converge")
    parser.add_argument("-d", "--data", dest="data", default="data.txt", type=str, help="training data")
    args = parser.parse_args()
    main(args)

