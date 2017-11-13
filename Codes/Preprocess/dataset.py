import numpy as np
import cPickle
import time, os

class InstanceBag(object):

    def __init__(self, entities, rel, num, sentences, positions, entitiesPos):
        self.entities = entities
        self.rel = rel
        self.num = num
        self.sentences = sentences
        self.positions = positions
        self.entitiesPos = entitiesPos


class DocumentContainer(object):
    def __init__(self, entity_pair, sentences, label,pos,l_dist,r_dist,entity_pos):
        self.entity_pair = entity_pair
        self.sentences = sentences
        self.label = label
        self.pos = pos
        self.l_dist = l_dist
        self.r_dist = r_dist
        self.entity_pos = entity_pos

def readData(filename, mode):
    print filename
    f = open(filename, 'r')
    data = []
    while 1:
        line = f.readline()
        if not line:
            break
        #  print line
        if mode == 1:
            num = line.split("\t")[3].strip().split(",")
            rel_string = line.split("\t")[2].strip()
        else:
            num = line.split("\t")[2].strip().split(",")
            rel_string = ""
        ldist = []
        rdist = []
        sentences = []
        entitiesPos = []
        pos = []
        rels = []
        for i in range(0, len(num)):
            sent = f.readline().strip().split(',')
            #entities = map(string,sent[:2])
            if len(sent) > 6:
              print sent
              new_sent = []
              ent = None
              for s in sent:
                try:
                  a = int(s)
                  if ent is not None:
                    new_sent.append(ent)
                    ent = None
                  new_sent.append(s)
                except:
                  if s[0] == '_':
                    ent += "," + s
                  else:
                    if ent is not None:
                      new_sent.append(ent)
                    ent = s
              sent = new_sent[:]
              print sent

            entities = sent[:2]
            epos = map(int,sent[2:4])
            epos.sort()
            rels.append(int(sent[4]))
            sent = f.readline().strip().split(",")
            sentences.append([(x+1) for x in map(int, sent)])
            for x in sentences[-1]:
                if x > 114044:
                    print sentences[-1]
                    sys.exit()
            sent = f.readline().strip().split(",")
            ldist.append(map(int, sent))
            sent = f.readline().strip().split(",")
            rdist.append(map(int, sent))
            entitiesPos.append(epos)
            pos.append([0]*len(sentences[-1]))
        rels = list(set(rels))
        #if len(rels) > 1:
         #   print rels
        ins = DocumentContainer(entity_pair=entities, sentences=sentences, label=rels, pos=pos, l_dist=ldist, r_dist=rdist, entity_pos=entitiesPos)
        data += [ins]
    f.close()
    return data

def wv2pickle(filename='wv.txt', dim=50, outfile='Wv.p'):
    f = open(filename, 'r')
    allLines = f.readlines()
    f.close()
    #tmp = allLines[0]
    Wv = np.zeros((len(allLines)+1, dim))
    i = 1
    for line in allLines:
        #tmp = map(float, line.split(' '))
        #Wv[i, :] = tmp
        line = line.split("\t")[1].strip()[:-1]
        #print line
        Wv[i, :] = map(float, line.split(','))
        i += 1

    rng = np.random.RandomState(3435)
    #tmp = rng.uniform(low=-0.5, high=0.5, size=(1, dim))
    Wv[1, :] = rng.uniform(low=-0.5, high=0.5, size=(1, dim)) #my unknown embedding
     #save Wv
    f = open(outfile, 'w')
    cPickle.dump(Wv, f, -1)
    f.close()

def data2pickle(input, output, mode):
    data = readData(input, mode)
    f = open(output, 'w')
    cPickle.dump(data, f, -1)
    f.close()



if __name__ == "__main__":
    # data = readData('train_filtered_len_60_gap_40.data')
    wv2pickle('word2vec.txt', 50, 'Wv.p')
    data2pickle('bags_train.txt','train_temp.p',1)
    data2pickle('bags_test.txt','test_temp.p',0)

