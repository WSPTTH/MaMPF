from sklearn.cluster import KMeans
import numpy as np
import Markov.markov as MK
import time
from sklearn.preprocessing import MinMaxScaler


class SMarkovModel(object):

    def __init__(self, order=1):
        self.order = order
        self.markovs = None

    def fit(self, status, _):
        self.markovs = [MK.Markov(data, order=self.order) for data in status]

    def predict(self, status, _):
        res = []
        for dataset in status:
            res_dataset = []
            for seq in dataset:
                lab = np.argmax(np.array([e.predict(seq) for e in self.markovs]))
                res_dataset.append(lab)
            res.append(res_dataset)
        return res

    def predict_prob(self, status, _):
        res = []
        for dataset in status:
            res_dataset = []
            for seq in dataset:
                res_dataset.append(np.array([e.predict(seq) for e in self.markovs]))
            res.append(np.array(res_dataset))
        return res


class LMarkovModel(SMarkovModel):

    def __init__(self, order=1):
        SMarkovModel.__init__(self, order)

    def fit(self, _, length):
        self.markovs = [MK.LengthMarkov(data, order=self.order) for data in length]

    def predict(self, _, length):
        res = []
        for dataset in length:
            res_dataset = []
            for seq in dataset:
                lab = np.argmax(np.array([e.predict(seq) for e in self.markovs]))
                res_dataset.append(lab)
            res.append(res_dataset)
        return res


class SMarkovModelWithCluster(SMarkovModel):

    def __init__(self, n_cluster=10, order=1):
        SMarkovModel.__init__(self, order)
        self.n_cluster = n_cluster
        self.kmeans = None
        self.prob = None

    def _cluster(self, length):
        self.kmeans = [KMeans(n_clusters=self.n_cluster).fit(data) for data in length]
        self.prob = np.zeros((len(self.kmeans), self.n_cluster))
        for app_ind, app_kmeans in enumerate(self.kmeans):
            for sam_label in app_kmeans.labels_:
                self.prob[app_ind][sam_label] += 1
        self.prob /= self.prob.sum(axis=1).reshape(-1, 1)

    def _normalize(self, length):
        # map to [0, 1], x_1 = (x - x_{min}) / (x_{max} - x_{min})
        self.x_max = length.max(axis=0)
        self.x_min = length.min(axis=0)
        return (length - self.x_min) / (self.x_max - self.x_min)

    def _dropLength(self, length):
        return [data[data[:, 0] > 0] for data in length]

    def fit(self, status, length):
        # here length is needed for cluster, containing the cert length and first package
        self.markovs = [MK.Markov(data, order=self.order) for data in status]
        length = [np.array(data).reshape((len(data), -1)) for data in length]
        length = self._dropLength(length)
        # length = [self._normalize(np.array(len_data)) for len_data in length]
        self._cluster(length)

    def _clusterPredict(self, length):
        c_p = np.zeros((len(length), len(self.kmeans)))
        for kx, kmeans in enumerate(self.kmeans):
            c_p[:, kx] = self.prob[kx][kmeans.predict(length)]
        return c_p

    def predict(self, status, length):
        res = []
        length = [np.array(data).reshape((len(data), -1)) for data in length]
        for sd, ld in zip(status, length):
            s_p = np.zeros((len(sd), len(status)))
            for ix, seq in enumerate(sd):
                s_p[ix] = np.array([e.predict(seq) for e in self.markovs])
            # ld = (np.array(ld) - self.x_min) / (self.x_max - self.x_min)
            c_p = self._clusterPredict(np.array(ld))
            for ix in range(len(ld)):
                if ld[ix][0] <= 0:
                    c_p[ix] = np.ones_like(c_p[ix])
            s_p *= c_p
            res.append(s_p.argmax(axis=1))
        return res


class SLMarkovModel(SMarkovModel):

    def __init__(self, order=1):
        SMarkovModel.__init__(self, order)
        self.smarkovs = None
        self.lmarkovs = None

    def fit(self, status, length):
        self.smarkovs = [MK.Markov(data, order=self.order) for data in status]
        self.lmarkovs = [MK.LengthMarkov(data, order=self.order) for data in length]

    def predict(self, status, length):
        res = []
        for sd, ld in zip(status, length):
            res_dataset = []
            for sseq, lseq in zip(sd, ld):
                sx = np.array([e.predict(sseq) for e in self.smarkovs])
                lx = np.array([e.predict(lseq) for e in self.lmarkovs])
                lab = np.argmax(sx * lx)
                res_dataset.append(lab)
            res.append(res_dataset)
        return res


class SMarkovClassify(object):

    def __init__(self, classifier, order=1):
        self.order = order
        self.markovs = None
        self.clf = classifier

    def _getMarkov(self, status, length):
        self.markovs = [MK.Markov(data, order=self.order) for data in status]

    def _normalize(self, vec, seq=None):
        vec = np.array(vec)
        a = np.sum(vec)
        if a == 0:
            return np.zeros_like(vec)
        # ---------- average ---------
        # vec /= a
        # ---------- largest is 1 --------
        # m = np.argmax(vec)
        # vec = np.zeros_like(vec)
        # vec[m] = 1
        # ---------- softmax
        # vec = np.exp(vec)
        # vec /= np.sum(vec)
        # ---------- max
        # vec /= np.max(vec)
        # vec = np.exp(vec)
        # vec /= np.sum(vec)
        # ======================================
        if seq is not None:
            vec = vec ** (1 / (len(seq) + 2 - self.order))
            # vec = np.exp(vec)
            # vec /= np.sum(vec)
        return vec

    def _getAppdata(self, app_data, markov):
        feature = []
        for seq in app_data:
            fe = [e.predict(seq) for e in markov]
            feature.append(self._normalize(fe, seq))
        return feature

    def _getfeature(self, data, markov):
        feature = []
        label = []
        for app_ind, app_data in enumerate(data):
            label.append(np.ones(len(app_data)) * app_ind)
            feature.append(self._getAppdata(app_data, markov))
        return np.concatenate(feature), np.concatenate(label)

    def fit(self, status, length):
        self._getMarkov(status, length)
        featrue, label = self._getfeature(status, self.markovs)
        self.clf.fit(featrue, label)

    def predict(self, status, length):
        res = []
        for dataset in status:
            app_featrue = self._getAppdata(dataset, self.markovs)
            res.append(self.clf.predict(app_featrue))
        return res


class LMarkovClassify(SMarkovClassify):

    def __init__(self, classifier, order=1):
        SMarkovClassify.__init__(self, classifier, order)

    def _getMarkov(self, status, length):
        self.markovs = [MK.LengthMarkov(data, order=self.order) for data in length]

    def fit(self, status, length):
        self._getMarkov(status, length)
        featrue, label = self._getfeature(length, self.markovs)
        self.clf.fit(featrue, label)

    def predict(self, status, length):
        res = []
        for dataset in length:
            app_featrue = self._getAppdata(dataset, self.markovs)
            res.append(self.clf.predict(app_featrue))
        return res


class SLMarkovClassify(SMarkovClassify):

    def __init__(self, classifier, order=1, lenPro=0.9):
        SMarkovClassify.__init__(self, classifier, order)
        self.smarkovs = None
        self.lmarkovs = None
        self.lenPro = lenPro

    def _getMarkov(self, status, length):
        self.smarkovs = [MK.Markov(data, order=self.order) for data in status]
        self.lmarkovs = [MK.LengthMarkov(data, order=self.order, prob=self.lenPro) for data in length]

    def _getSeqFeature(self, seq, markov):
        return self._normalize([e.predict(seq) for e in markov], seq)

    def _getAppdata(self, data, markovs):
        feature = []
        status, length = data
        smarkov, lmarkov = markovs
        for sseq, lseq in zip(status, length):
            feature.append(self._getSeqFeature(sseq, smarkov) + self._getSeqFeature(lseq, lmarkov))
        return np.array(feature)

    def _getfeature(self, data, markov):
        feature = []
        label = []
        for app_ind, app_data in enumerate(data):
            label.append(np.ones(len(app_data[0])) * app_ind)
            feature.append(self._getAppdata(app_data, markov))
        return np.concatenate(feature), np.concatenate(label)

    def fit(self, status, length):
        self._getMarkov(status, length)
        data = [(apps, appl) for apps, appl in zip(status, length)]
        featrue, label = self._getfeature(data, (self.smarkovs, self.lmarkovs))
        self.clf.fit(featrue, label)
        return self

    def predict(self, status, length):
        res = []
        for sd, ld in zip(status, length):
            app_feature = self._getAppdata((sd, ld), (self.smarkovs, self.lmarkovs))
            res.append(self.clf.predict(app_feature))
        return res


if __name__ == '__main__':
    pass
