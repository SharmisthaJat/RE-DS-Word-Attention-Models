from dataset import *
import time
import cPickle
from random import shuffle

class DocumentContainer(object):
    def __init__(self, entity_pair, sentences, label,pos,l_dist,r_dist,entity_pos):
        self.entity_pair = entity_pair
        self.sentences = sentences
        self.label = label
        self.pos = pos
        self.l_dist = l_dist
        self.r_dist = r_dist
        self.entity_pos = entity_pos


def get_ins(snum, index1, index2, pos, filter_h=5, max_l=100):
    pad = int(filter_h/2)
    x = [0]*pad
    pf1 = [0]*pad
    pf2 = [0]*pad
    if pos[0] == pos[1]:
        #print pos
        if (pos[1] + 1) < len(snum):
            pos[1] = pos[1] + 1
        else:
            pos[0] = pos[0] - 1
        #print pos
    if len(snum) <= max_l:
        for i, ind in enumerate(snum):
            x.append(ind)
            pf1.append(index1[i] + 1)
            pf2.append(index2[i] + 1)
    else:
        idx = [q for q in range(pos[0], pos[1] + 1)]
        if len(idx) > max_l:
            idx = idx[:max_l]
            for i in idx:
                x.append(snum[i])
                pf1.append(index1[i] + 1)
                pf2.append(index2[i] + 1)
            # print snum, index1, index2, pos
            # sys.exit()
            pos[0] = 0
            pos[1] = len(idx) - 1
        else:
            for i in idx:
                x.append(snum[i])
                pf1.append(index1[i] + 1)
                pf2.append(index2[i] + 1)

            before = pos[0] - 1
            after = pos[1] + 1
            pos[0] = 0
            pos[1] = len(idx) - 1
            #print before, after, pos, idx[0], idx[-1]
            numAdded = 0
            while True:
                added = 0
                if before >= 0 and (len(x) + 1) <= (max_l+pad):
                    x.append(snum[before])
                    pf1.append(index1[before] + 1)
                    pf2.append(index2[before] + 1)
                    added = 1
                    numAdded += 1

                if after < len(snum) and (len(x) + 1) <= (max_l+pad):
                    x.append(snum[after])
                    pf1.append(index1[after] + 1)
                    pf2.append(index2[after] + 1)
                    added = 1
                if added == 0:
                    break
                before = before - 1
                after = after + 1

            pos[0] = pos[0] + numAdded
            pos[1] = pos[1] + numAdded
        if len(x) != (max_l+1):
            print x
            print len(x)
            sys.exit()
    while len(x) < max_l+2*pad:
        x.append(0)
        pf1.append(0)
        pf2.append(0)

    return [x,pf1,pf2,pos]

def get_idx(snum, filter_h=5, max_l=100):

    pad = int(filter_h/2)
    x = [0]*pad
    if len(snum) < max_l:
        for ind in snum:
            if ind == 0:
                print snum
                sys.exit()
            x.append(ind)
    else:
        for i in xrange(max_l):
            if snum[i] == 0:
                print snum
                sys.exit()
            x.append(snum[i])
    #padding the end of sentence
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def get_pf(index1, filter_h=5, max_l=100):
    pad = int(filter_h/2)
    pf1 = [0]*pad
    if len(index1) < max_l:
        for ind in index1:
            ind = ind + 1
            if ind == 0:
                print index1
                sys.exit()
            pf1.append(ind)
    else:
        for i in xrange(max_l):
            index1[i] = index1[i] + 1
            if index1[i] == 0:
                print index1
                sys.exit()
            pf1.append(index1[i])
    while len(index1) < max_l+2*pad:
        pf1.append(0)

    return pf1

def make_idx_data_cv(data, filter_h, max_l):
    newData = []
    for j,ins in enumerate(data):
        entities = ins.entity_pair
        rel = ins.label
        pos = ins.pos
        sentences = ins.sentences
        ldist = ins.l_dist
        rdist = ins.r_dist
        newSent = []
        l_dist = []
        r_dist = []
        entitiesPos = ins.entity_pos
        newent = []
        # print j

        for i, sentence in enumerate(sentences):
            idx,a,b,e = get_ins(sentence, ldist[i], rdist[i], entitiesPos[i], filter_h, max_l)
            for qq in idx:
                if qq > 114044:
                    print idx
                    print j
                    sys.exit()
            newSent.append(idx[:])
            l_dist.append(a[:])
            r_dist.append(b[:])
            newent.append(e[:])
        # newIns = InstanceBag(entity_pair=entities, sentences=newSent, label=rel, newSent, newPos, entitiesPos)
        newIns = DocumentContainer(entity_pair=entities, sentences=newSent, label=rel, pos=pos, l_dist=l_dist, r_dist=r_dist, entity_pos=newent)
        newData += [newIns]
    return newData


if __name__ == "__main__":
    print "load test and train raw data..."

    testData = cPickle.load(open('test_temp.p'))
    trainData = cPickle.load(open('train_temp.p'))
    devData = cPickle.load(open('dev_temp.p'))
    sentence_len = 80
    max_filter_len = 3
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print 'point 0 time: ' + '\t\t' + str(now)
    test = make_idx_data_cv(testData, max_filter_len, sentence_len)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print 'point 1 time: ' + '\t\t' + str(now)
    train = make_idx_data_cv(trainData, max_filter_len, sentence_len)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print 'point 2 time: ' + '\t\t' + str(now)
    dev = make_idx_data_cv(devData, max_filter_len, sentence_len)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print 'point 3 time: ' + '\t\t' + str(now)

    print "Len train:",len(train)
    print "Len dev:",len(dev)
    f = open('test_final.p','w')
    cPickle.dump(test, f, -1)
    f.close()

    f = open('train_final.p', 'w')
    cPickle.dump(train, f, -1)
    f.close()
    
    f = open('dev_final.p', 'w')
    cPickle.dump(dev, f, -1)
    f.close()
