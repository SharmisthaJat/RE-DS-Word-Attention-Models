import os, sys
import cPickle
import math
import time
sys.path.insert(0, '/home/mall/.local/lib/python2.7/site-packages/')
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as Fh
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as weight_init
import gc
import operator
import random
from sklearn.metrics import average_precision_score
import getopt

print "Please Note That CUDA is required for the model to run"
use_cuda = torch.cuda.is_available()
print use_cuda
gpu = 1

def parse_argv(argv):
    opts, args = getopt.getopt(sys.argv[1:], "he:i:o:m:",
                               ['epoch','input','output','mode'])
    epochs = 50
    output = './'
    mode = 1
    inputs = ""
    for op, value in opts:
        print op,value
        if op == '-e':
            epochs = int(value)
        elif op == '-o':
            output = value
        elif op == '-m':
            mode =int(value)
        elif op == '-i':
            inputs = value
        elif op == '-h':
            #TODO
            #usage()
            sys.exit()
    return [epochs, inputs, output, mode]

class DocumentContainer(object):
    def __init__(self, entity_pair, sentences, label,pos,l_dist,r_dist,entity_pos):
        self.entity_pair = entity_pair
        self.sentences = sentences
        self.label = label
        self.pos = pos
        self.l_dist = l_dist
        self.r_dist = r_dist
        self.entity_pos = entity_pos

class getEmbeddings(nn.Module):
	def __init__(self, word_size, word_length, feature_size, feature_length, Wv, pf1, pf2):
		super(getEmbeddings, self).__init__()
		self.x_embedding = nn.Embedding(word_length, word_size, padding_idx=0)
		self.ldist_embedding = nn.Embedding(feature_length, feature_size, padding_idx=0)
		self.rdist_embedding = nn.Embedding(feature_length, feature_size, padding_idx=0)
		self.x_embedding.weight.data.copy_(torch.from_numpy(Wv))
		self.ldist_embedding.weight.data.copy_(torch.from_numpy(pf1))
		self.rdist_embedding.weight.data.copy_(torch.from_numpy(pf2))

	def forward(self, x, ldist, rdist):
		x_embed = self.x_embedding(x)
		ldist_embed = self.ldist_embedding(ldist)
		rdist_embed = self.rdist_embedding(rdist)
		concat = torch.cat([x_embed, ldist_embed, rdist_embed], x_embed.dim() - 1)
		return concat.unsqueeze(1)

class GRU(nn.Module):
    def __init__(self, input_dim, gru_layers, input_words):
    	super(GRU,self).__init__()
        self.input_dim = input_dim
        self.gru_layers = gru_layers
        self.input_words = input_words
    	self.gru = nn.GRU(input_dim, gru_layers, 1, batch_first=True, bidirectional=True, bias=True)

    def forward(self, x):
        mask = (1. - torch.eq(x,0.).float()).narrow(x.dim() -1,0,1).squeeze(1).expand(x.size(0),x.size(2), self.gru_layers*2)
        x = x.squeeze(1)
        hidden = self.init_hidden(x.size(0))
        gru, hidden = self.gru(x, hidden)
        gru = gru*mask
        gru = gru.contiguous()
        return gru, mask

    def init_hidden(self, size):
        hidden = autograd.Variable(torch.zeros(2, size, self.gru_layers))
        if use_cuda:
            hidden = hidden.cuda(gpu)
        return hidden

class WordAttention(nn.Module):
    def __init__(self, input_dim, gru_layers, input_words):
        super(WordAttention,self).__init__()
        self.input_dim = input_dim
        self.gru_layers = gru_layers
        self.input_words = input_words
        self.relationMatrix = nn.Linear(input_words*gru_layers*2, gru_layers*2, bias=False)
        self.relationVector = nn.Linear(gru_layers*2, input_words, bias=False)

    def forward(self, gru, mask):
        attention_values = self.relationVector(F.tanh(self.relationMatrix(gru.view(gru.size(0),-1)))).view((gru.size(0), self.input_words)).unsqueeze(2).expand(gru.size(0), self.input_words, self.gru_layers*2)
        # attention_values = self.relationVector(F.tanh(gru.view(gru.size(0),-1))).view((gru.size(0), self.input_words)).unsqueeze(2).expand(gru.size(0), self.input_words, self.gru_layers*2)
        attention_exp = torch.exp(attention_values) * mask
        attention_values_sum = torch.sum(attention_exp,1).expand(gru.size(0), self.input_words, self.gru_layers*2)
        attention_values_softmax = (attention_exp/attention_values_sum)
        attention_embeddings_softmax = gru * attention_values_softmax
        # attention_sentence_embedding = torch.sum(attention_embeddings_softmax,1).view((-1, self.gru_layers*2))
        return attention_embeddings_softmax

class WordAttentionSimple(nn.Module):
    def __init__(self, input_dim, gru_layers, input_words):
        super(WordAttentionSimple,self).__init__()
        self.input_dim = input_dim
        self.gru_layers = gru_layers
        self.input_words = input_words
        self.relationMatrix = nn.Linear(gru_layers*2, gru_layers*2, bias=False)
        self.relationVector = nn.Linear(gru_layers*2, 1, bias=False)

    def forward(self, gru, mask):
        attention_values = self.relationVector(F.tanh(self.relationMatrix(gru.view(-1,gru.size(2))))).view((gru.size(0), self.input_words)).unsqueeze(2).expand(gru.size(0), self.input_words, self.gru_layers*2)
        # attention_values = self.relationVector(F.tanh(gru.view(gru.size(0),-1))).view((gru.size(0), self.input_words)).unsqueeze(2).expand(gru.size(0), self.input_words, self.gru_layers*2)
        attention_exp = torch.exp(attention_values) * mask
        attention_values_sum = torch.sum(attention_exp,1).expand(gru.size(0), self.input_words, self.gru_layers*2)
        attention_values_softmax = (attention_exp/attention_values_sum)
        attention_embeddings_softmax = gru * attention_values_softmax
        # attention_sentence_embedding = torch.sum(attention_embeddings_softmax,1).view((-1, self.gru_layers*2))
        return attention_embeddings_softmax, attention_values_softmax

class PieceWisePool(nn.Module):
    def __init__(self):
        super(PieceWisePool,self).__init__()

    def forward(self, gru, entity_pos):
        concat_list = []
        # entity_pos = entity_pos.data
        for index, entity in enumerate(entity_pos):
            elem = gru.narrow(0,index,1)
            pool1 = F.max_pool2d(elem.narrow(1,0,entity[0]),(entity[0],1))
            pool2 = F.max_pool2d(elem.narrow(1,entity[0],entity[1]-entity[0]),(entity[1]-entity[0],1))
            pool3 = F.max_pool2d(elem.narrow(1,entity[1],gru.size(1)-entity[1]),(gru.size(1)-entity[1],1))
            concat_pool = torch.cat((pool1, pool2, pool3), gru.dim()-1)
            concat_list.append(concat_pool.squeeze(1))
        concat_all = torch.cat(concat_list,0)
        return concat_all

class SumAttention(nn.Module):
    def __init__(self):
        super(SumAttention, self).__init__()
        
    def forward(self, gru):
        return torch.sum(gru, 1).squeeze(1)

class CNNwithPool(nn.Module):
    def __init__(self, cnn_layers, kernel_size):
    	super(CNNwithPool,self).__init__()
    	self.cnn = nn.Conv2d(1, cnn_layers, kernel_size)
    	self.cnn.bias.data.copy_(weight_init.constant(self.cnn.bias.data,0.))

    def forward(self, x, entity_pos):
        cnn = self.cnn(x)
        concat_list = []
        for index, entity in enumerate(entity_pos):
            elem = cnn.narrow(0,index,1)
            pool1 = F.max_pool2d(elem.narrow(2,0,entity[0]),(entity[0],1))
            pool2 = F.max_pool2d(elem.narrow(2,entity[0],entity[1]-entity[0]),(entity[1]-entity[0],1))
            pool3 = F.max_pool2d(elem.narrow(2,entity[1],cnn.size(2)-entity[1]),(cnn.size(2)-entity[1],1))
            concat_pool = torch.cat((pool1, pool2, pool3), cnn.dim()-1)
            concat_list.append(concat_pool)
        concat_all = torch.cat(concat_list,0)
        return concat_all

class PCNN(nn.Module):
    def __init__(self, word_length, feature_length, cnn_layers, Wv, pf1, pf2, kernel_size, word_size=50, feature_size=5, dropout=0.5, num_classes=53, num_words=82):
        super(PCNN, self).__init__()
        self.word_length = word_length
        self.feature_length = feature_length
        self.cnn_layers = cnn_layers
        self.kernel_size = kernel_size
        self.word_size = word_size
        self.feature_size = feature_size
        self.num_classes = num_classes

        self.embeddings = getEmbeddings(self.word_size, self.word_length, self.feature_size, self.feature_length, Wv, pf1, pf2)
        self.gru = GRU(self.word_size + 2*self.feature_size, self.cnn_layers, num_words)
        self.wordAttention = WordAttentionSimple(self.word_size + 2*self.feature_size, self.cnn_layers, num_words)
        self.pieceWisePool = PieceWisePool()
        # self.word_attention = CommonWordAttention(self.num_classes, self.cnn_layers*2)
        # self.cnn = CNNwithPool(self.cnn_layers, self.kernel_size)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(self.cnn_layers*6, self.num_classes)

    def forward(self, x, ldist, rdist, pool):
        embeddings = self.embeddings(x,ldist,rdist)
        gru, mask = self.gru(embeddings)
        wordAttention, attention_scores = self.wordAttention(gru,mask)  
        pieceWisePool = self.pieceWisePool(wordAttention, pool)
        #pieceWisePool = self.pieceWisePool(wordAttention)
        sentence_embedding = F.tanh(pieceWisePool)
        cnn_dropout = self.drop(sentence_embedding)
        probabilities = self.linear(cnn_dropout)
        return probabilities, [attention_scores]

def trainModel(train, test, dev, epochs, directory, Wv, pf1, pf2, batch=50, num_classes=53, max_sentences=5, img_h=82, to_train=1, test_epoch=0):
    model = PCNN(word_length=len(Wv), feature_length=len(pf1), cnn_layers=230, kernel_size=(3,60), Wv=Wv, pf1=pf1, pf2=pf2, num_classes=num_classes)

    # model = nn.DataParallel(model, device_ids=[0,1]).cuda(gpu)
    model = model.cuda(gpu)
    [test_label, test_sents, test_pos, test_ldist, test_rdist, test_entity, test_epos] = bags_decompose(test)
    [dev_label, dev_sents, dev_pos, dev_ldist, dev_rdist, dev_entity, dev_epos] = bags_decompose(dev)
    [train_label, train_sents, train_pos, train_ldist, train_rdist, train_entity, train_epos] = bags_decompose(train)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # optimizer = optim.Adam(model.parameters())
    if test_epoch != 0 and to_train==1:
        print "Loading:","model_"+str(test_epoch)
        checkpoint = torch.load(directory+"model_"+str(test_epoch), map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    loss_function = nn.CrossEntropyLoss().cuda(gpu)
    totalBatches = int(math.ceil(len(train_label)/batch))
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print "Training:",str(now)
    for epoch in range(test_epoch,epochs):
        if to_train == 1:
            model.eval()
            train_data, train_labels, train_poss, train_ldists, train_rdists, train_eposs = select_instance3(train_label, train_sents, train_pos, train_ldist, train_rdist, train_epos, img_h, num_classes, max_sentences, model, batch=2000)
            total_loss = 0.
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            samples = train_data.shape[0]
            batches = _make_batches(samples, batch)
            index_array = np.arange(samples)
            random.shuffle(index_array)
            print str(now),"\tStarting Epoch",(epoch),"\tBatches:",len(batches)
            model.train()
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                x_slice = torch.from_numpy(_slice_arrays(train_data, batch_ids)).long().cuda(gpu)
                l_slice = torch.from_numpy(_slice_arrays(train_ldists, batch_ids)).long().cuda(gpu)
                r_slice = torch.from_numpy(_slice_arrays(train_rdists, batch_ids)).long().cuda(gpu)
                e_slice = torch.from_numpy(_slice_arrays(train_eposs, batch_ids)).long().cuda(gpu)
                train_labels_slice = torch.from_numpy(_slice_arrays(train_labels, batch_ids)).long().cuda(gpu)
                x_batch = autograd.Variable(x_slice)
                l_batch = autograd.Variable(l_slice)
                r_batch = autograd.Variable(r_slice)
                e_batch = e_slice
                train_labels_batch = autograd.Variable(train_labels_slice)
                results_batch, attention_scores = model(x_batch, l_batch, r_batch, e_batch)
                loss = loss_function(results_batch, train_labels_batch)
                optimizer.zero_grad()
                total_loss += loss.data
                loss.backward()
                # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
                optimizer.step()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print str(now),"\tDone Epoch",(epoch),"\tLoss:",total_loss
            torch.save({'epoch': epoch ,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}, directory+"model_"+str(epoch))

            model.eval()
            dev_predict = get_test3(dev_label, dev_sents, dev_pos, dev_ldist, dev_rdist, dev_epos, img_h, num_classes, max_sentences, model, batch=2000)

            if to_train == 1:
                cPickle.dump(dev_predict,open(directory+"predict_prob_dev_"+str(epoch),"wb"))
            else:
                cPickle.dump(dev_predict,open(directory+"predict_prob_dev_temp_"+str(epoch),"wb"))
            print "Test"

            dev_pr = pr(dev_predict[3], dev_predict[2], dev_entity)
            accuracy(dev_predict[3], dev_predict[2])
            one_hot = []
            results = dev_predict[3]
            for labels in dev_label:
                arr = np.zeros(shape=(num_classes-1,),dtype='int32')
                for label in labels:
                    if label != 0:
                        arr[label-1] = 1
                one_hot.append(arr)
            one_hot = np.array(one_hot)
            results = results[:,1:]
            score = average_precision_score(one_hot, results, average='micro')
            if to_train == 1:
                out = open(directory+"pr_dev_"+str(epoch),"wb")
            else:
                out = open(directory+"pr_dev_temp_"+str(epoch),"wb")
            for res in dev_pr[3]:
                out.write(str(res[0])+"\t"+str(res[1])+"\n")
            out.close()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            precision = -1
            recall = -1
            print str(now) + '\t epoch ' + str(epoch) + "\tTest\tScore:"+str(score)+"\t Precision : "+str(dev_pr[0]) + "\t Recall: "+str(dev_pr[1])+ "\t Total: "+ str(dev_pr[2])
        else:
            print "Loading:","model_"+str(epoch)
            checkpoint = torch.load(directory+"model_"+str(epoch), map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])    
        
            test_predict = get_test3(test_label, test_sents, test_pos, test_ldist, test_rdist, test_epos, img_h, num_classes, max_sentences, model, batch=2000)
            # test_predict = visualize_attention(test_label, test_sents, test_pos, test_ldist, test_rdist, test_epos, img_h, num_classes, max_sentences, model, batch=2000)
            if to_train == 1:
                cPickle.dump(test_predict,open(directory+"predict_prob_"+str(epoch),"wb"))
            else:
                cPickle.dump(test_predict,open(directory+"predict_prob_temp_"+str(epoch),"wb"))
            print "Test"

            test_pr = pr(test_predict[3], test_predict[2], test_entity)
            accuracy(test_predict[3], test_predict[2])
            one_hot = []
            results = test_predict[3]
            for labels in test_label:
                arr = np.zeros(shape=(num_classes-1,),dtype='int32')
                for label in labels:
                    if label != 0:
                        arr[label-1] = 1
                one_hot.append(arr)
            one_hot = np.array(one_hot)
            results = results[:,1:]
            score = average_precision_score(one_hot, results, average='micro')
            if to_train == 1:
                out = open(directory+"pr_"+str(epoch),"wb")
            else:
                out = open(directory+"pr_temp_"+str(epoch),"wb")
            for res in test_pr[3]:
                out.write(str(res[0])+"\t"+str(res[1])+"\n")
            out.close()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            precision = -1
            recall = -1
            print str(now) + '\t epoch ' + str(epoch) + "\tTest\tScore:"+str(score)+"\t Precision : "+str(test_pr[0]) + "\t Recall: "+str(test_pr[1])+ "\t Total: "+ str(test_pr[2])
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print str(now) + '\t epoch ' + str(epoch) + ' save PR result...'
            print '\n'

def bags_decompose(data_bags):
    bag_sent = [data_bag.sentences for data_bag in data_bags]
    bag_label = [data_bag.label for data_bag in data_bags]
    bag_pos = [data_bag.pos for data_bag in data_bags]
    bag_ldist = [data_bag.l_dist for data_bag in data_bags]
    bag_rdist = [data_bag.r_dist for data_bag in data_bags]
    bag_entity = [data_bag.entity_pair for data_bag in data_bags]
    bag_epos = [data_bag.entity_pos for data_bag in data_bags]
    return [bag_label, bag_sent, bag_pos, bag_ldist, bag_rdist, bag_entity, bag_epos]

def accuracy(predict_y, true_y):
    correct = 0
    count = 0
    for i,label in enumerate(true_y):
        if len(true_y[i]) ==1 and true_y[i][0] == 0:
            continue
        else:
            count += 1
            if np.argmax(predict_y[i]) in true_y[i]:
                correct += 1
    print "accuracy: ",float(correct)/count, correct, count
    
def pr(predict_y, true_y,entity_pair):
    final_labels = []
    for label in true_y:
        if 0 in label and len(label) > 1:
            label = [x for x in label if x!=0]
        final_labels.append(label[:])

    total = 0
    for label in final_labels:
        if 0 in label:
            continue
        else:
            total += len(label)
    print "Total:",total
    results = []
    for i in range(predict_y.shape[0]):
        for j in range(1, predict_y.shape[1]):
            results.append([i,j,predict_y[i][j],entity_pair[i]])
    resultSorted = sorted(results, key=operator.itemgetter(2),reverse=True)

    p_p = 0.0
    p_n = 0.0
    n_p = 0.0
    pr = []
    prec = 0.0
    rec = 0.0
    p_p_final = 0.0
    p_n_final = 0.0
    n_p_final = 0.0
    prev = -1
    for g,item in enumerate(resultSorted):
        prev = item[2]
        if 0 in final_labels[item[0]]:
            if item[1] == 0:
                temp = 1
            else:
                n_p += 1
        else:
            if item[1] in final_labels[item[0]]:
                p_p += 1
            else:
                p_n += 1
        # if g%100 == 0:
            # print "Precision:",(p_p)/(p_p+n_p)
            # print "Recall",(p_p)/total

        try:
            pr.append([(p_p)/(p_p+n_p+p_n), (p_p)/total])
        except:
            pr.append([1.0,(p_p)/total])
        if rec <= 0.3:
            try:
                prec = (p_p)/(p_p+n_p+p_n)
            except:
                prec = 1.0
            rec = (p_p)/total
            p_p_final = p_p
            p_n_final = p_n
            n_p_final = n_p

    print "p_p:",p_p_final,"n_p:",n_p_final,"p_n:",p_n_final
    return [prec,rec,total,pr]

def _make_batches(size, batch_size):
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]

def _slice_arrays(arrays, start=None, stop=None):
    if isinstance(arrays, list):
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in arrays]
        else:
            return [x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        else:
            return arrays[start:stop]


def get_test3(label, sents, pos, ldist, rdist, epos, img_h , numClasses, maxSentences, testModel, filterSize = 3, batch = 1000):
    numBags = len(label)
    predict_y = np.zeros((numBags), dtype='int32')
    predict_y_prob = np.zeros((numBags), dtype='float32')
    predict_y_dist = np.zeros((numBags, numClasses), dtype='float32')
    # y = np.asarray(rels, dtype='int32')
    numSentences = 0
    for ind in range(len(sents)):
        numSentences += len(sents[ind])
    print "Num Sentences:", numSentences
    insX = np.zeros((numSentences, img_h), dtype='int32')
    insPf1 = np.zeros((numSentences, img_h), dtype='int32')
    insPf2 = np.zeros((numSentences, img_h), dtype='int32')
    insPool = np.zeros((numSentences, 2), dtype='int32')
    currLine = 0
    for bagIndex, insRel in enumerate(label):
        insNum = len(sents[bagIndex])
        for m in range(insNum):
            insX[currLine] = np.asarray(sents[bagIndex][m], dtype='int32').reshape((1, img_h))
            insPf1[currLine] = np.asarray(ldist[bagIndex][m], dtype='int32').reshape((1, img_h))
            insPf2[currLine] = np.asarray(rdist[bagIndex][m], dtype='int32').reshape((1, img_h))
            epos[bagIndex][m] = sorted(epos[bagIndex][m])
            if epos[bagIndex][m][0] > 79:
                epos[bagIndex][m][0] = 79
            if epos[bagIndex][m][1] > 79:
                epos[bagIndex][m][1] = 79
            if epos[bagIndex][m][0] == epos[bagIndex][m][1]:
                insPool[currLine] = np.asarray([epos[bagIndex][m][0]+int(filterSize/2), epos[bagIndex][m][1]+ int(filterSize/2) + 1], dtype='int32').reshape((1, 2))
            else:
                insPool[currLine] = np.asarray([epos[bagIndex][m][0]+int(filterSize/2), epos[bagIndex][m][1]+ int(filterSize/2)], dtype='int32').reshape((1, 2))
            currLine += 1
    insX = np.array(insX.tolist())
    insPf1 = np.array(insPf1.tolist())
    insPf2 = np.array(insPf2.tolist())
    insPool = np.array(insPool.tolist())
    results = []
    totalBatches = int(math.ceil(float(insX.shape[0])/batch))
    results = np.zeros((numSentences, numClasses), dtype='float32')
    print "totalBatches:",totalBatches
    samples = insX.shape[0]
    batches = _make_batches(samples, batch)
    index_array = np.arange(samples)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        # print batch_index, (batch_start, batch_end)
        batch_ids = index_array[batch_start:batch_end]
        x_batch = autograd.Variable(torch.from_numpy(_slice_arrays(insX, batch_ids)).long().cuda(gpu), volatile=True)
        l_batch = autograd.Variable(torch.from_numpy(_slice_arrays(insPf1, batch_ids)).long().cuda(gpu), volatile=True)
        r_batch = autograd.Variable(torch.from_numpy(_slice_arrays(insPf2, batch_ids)).long().cuda(gpu), volatile=True)
        # e_batch = autograd.Variable(torch.from_numpy(_slice_arrays(insPool, batch_ids)).long().cuda(gpu), volatile=True)
        e_batch = torch.from_numpy(_slice_arrays(insPool, batch_ids)).long().cuda(gpu)
        results_batch, attention_scores = testModel(x_batch, l_batch, r_batch, e_batch)
        results[batch_start:batch_end,:] = F.softmax(results_batch).data.cpu().numpy()
    # print results
    rel_type_arr = np.argmax(results,axis=-1)
    max_prob = np.amax(results, axis=-1)
    currLine = 0
    for bagIndex, insRel in enumerate(label):
        insNum = len(sents[bagIndex])
        maxP = -1
        pred_rel_type = 0
        max_pos_p = -1
        positive_flag = False
        max_vec = []

        for m in range(insNum):
            rel_type = rel_type_arr[currLine]
            if positive_flag and rel_type == 0:
                currLine += 1
                continue
            else:
                # at least one instance is positive
                tmpMax = max_prob[currLine]
                if rel_type > 0:
                    positive_flag = True
                    if tmpMax > max_pos_p:
                        max_pos_p = tmpMax
                        pred_rel_type = rel_type
                        max_vec = np.copy(results[currLine])
                else:
                    if tmpMax > maxP:
                        maxP = tmpMax
                        max_vec = np.copy(results[currLine])
                currLine += 1
        if positive_flag:
            predict_y_prob[bagIndex] = max_pos_p
        else:
            predict_y_prob[bagIndex] = maxP
        predict_y_dist[bagIndex] =  np.asarray(np.copy(max_vec), dtype='float32').reshape((1,numClasses))
        predict_y[bagIndex] = pred_rel_type
    return [predict_y, predict_y_prob, label, predict_y_dist]

def select_instance3(label, sents, pos, ldist, rdist, epos, img_h , numClasses, maxSentences, testModel, filterSize = 3,batch=1000):
    numBags = len(label)
    y_final = []
    xL = np.zeros((numBags, img_h), dtype='int32')
    pL = np.zeros((numBags, img_h), dtype='int32')
    lL = np.zeros((numBags, img_h), dtype='int32')
    rL = np.zeros((numBags, img_h), dtype='int32')
    eL = np.zeros((numBags, 2), dtype='int32')
    labL = np.zeros((numBags,1),dtype='int32')
    bagIndexX = 0
    totalSents = 0
    for bagIndex, insNum in enumerate(sents):
        totalSents += len(insNum)
    numSentences = {}
    x = np.zeros((totalSents, img_h), dtype='int32')
    p = np.zeros((totalSents, img_h), dtype='int32')
    l = np.zeros((totalSents, img_h), dtype='int32')
    r = np.zeros((totalSents, img_h), dtype='int32')
    e = np.zeros((totalSents, 2), dtype='int32')
    lab = np.zeros((totalSents, 1), dtype='int32')
    curr = 0
    for bagIndex, insNum in enumerate(sents):
        numSentences[bagIndex] = len(insNum)
        if len(insNum) > 0:
            bagNum = 0
            for m in range(len(insNum)):
                x[curr,:] = sents[bagIndex][m]
                l[curr,:] = ldist[bagIndex][m]
                r[curr,:] = rdist[bagIndex][m]
                lab[curr, :] = [label[bagIndex][0]]
                epos[bagIndex][m] = sorted(epos[bagIndex][m])
                if epos[bagIndex][m][0] > 79:
                    epos[bagIndex][m][0] = 79
                if epos[bagIndex][m][1] > 79:
                    epos[bagIndex][m][1] = 79
                if epos[bagIndex][m][0] == epos[bagIndex][m][1]:
                    e[curr,:] = [epos[bagIndex][m][0]+int(filterSize/2), epos[bagIndex][m][1]+ int(filterSize/2) + 1]
                else:
                    e[curr,:] = [epos[bagIndex][m][0]+int(filterSize/2), epos[bagIndex][m][1]+ int(filterSize/2)]
                curr += 1
    totalBatches = int(math.ceil(float(x.shape[0])/batch))
    results = []
    x = np.array(x.tolist())
    l = np.array(l.tolist())
    r = np.array(r.tolist())
    e = np.array(e.tolist())
    # print e, e.shape
    results = np.zeros((totalSents, numClasses), dtype='float32')
    print "totalBatches:",totalBatches
    samples = x.shape[0]
    batches = _make_batches(samples, batch)
    index_array = np.arange(samples)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        # print batch_index
        batch_ids = index_array[batch_start:batch_end]
        x_batch = autograd.Variable(torch.from_numpy(_slice_arrays(x, batch_ids)).long().cuda(gpu), volatile=True)
        l_batch = autograd.Variable(torch.from_numpy(_slice_arrays(l, batch_ids)).long().cuda(gpu), volatile=True)
        r_batch = autograd.Variable(torch.from_numpy(_slice_arrays(r, batch_ids)).long().cuda(gpu), volatile=True)
        # e_batch = autograd.Variable(torch.from_numpy(_slice_arrays(e, batch_ids)).long().cuda(gpu), volatile=True)
        e_batch = torch.from_numpy(_slice_arrays(e, batch_ids)).long().cuda(gpu)
        results_batch, attention_scores = testModel(x_batch, l_batch, r_batch, e_batch)
        results[batch_start:batch_end,:] = F.softmax(results_batch).data.cpu().numpy()

    predict_y_prob = np.amax(results, axis=-1)
    predict_y = np.argmax(results, axis=-1)
    curr = 0
    for bagIndex, insNum in enumerate(sents):
        maxp = -1
        max_ins = 0
        if len(insNum) > 0:
            bagNum = 0
            for m in range(len(insNum)):
                if predict_y_prob[curr] > maxp:
                    maxp = predict_y_prob[curr]
                    max_ins = m
                curr += 1
            xL[bagIndex, :] = sents[bagIndex][max_ins]
            lL[bagIndex, :] = ldist[bagIndex][max_ins]
            rL[bagIndex, :] = rdist[bagIndex][max_ins]
            labL[bagIndex, :] = [label[bagIndex][0]]
            epos[bagIndex][max_ins] = sorted(epos[bagIndex][max_ins])
            if epos[bagIndex][max_ins][0] > 79:
                epos[bagIndex][max_ins][0] = 79
            if epos[bagIndex][max_ins][1] > 79:
                epos[bagIndex][max_ins][1] = 79
            if epos[bagIndex][max_ins][0] == epos[bagIndex][max_ins][1]:
                eL[bagIndex,:] = [epos[bagIndex][max_ins][0]+int(filterSize/2), epos[bagIndex][max_ins][1]+ int(filterSize/2) + 1]
            else:
                eL[bagIndex,:] = [epos[bagIndex][max_ins][0]+int(filterSize/2), epos[bagIndex][max_ins][1]+ int(filterSize/2)]
            # y = np.zeros((numClasses,))
            # y[label[bagIndex][0]] = 1
            y_final.append(label[bagIndex][0])
    xL = np.array(xL.tolist())
    lL = np.array(lL.tolist())
    rL = np.array(rL.tolist())
    eL = np.array(eL.tolist())
    y_final = np.asarray(y_final, dtype='int32')
    return [xL,y_final,pL,lL,rL,eL]




if __name__ == "__main__":
    if len(sys.argv) < 6:
        print "Please enter the arguments correctly!"
        sys.exit()

    inputdir = sys.argv[1] + "/"
    resultdir = inputdir
    resultdir = "BGWA/"
    print 'result dir='+resultdir
    if not os.path.exists(resultdir):
        os.mkdir(resultdir)


    dataType = "_features_all_6Months"
    test = cPickle.load(open(inputdir+sys.argv[3]))
    train = cPickle.load(open(inputdir+sys.argv[2]))
    dev = cPickle.load(open(inputdir+sys.argv[4]))
    print 'load Wv ...'
    Wv = np.array(cPickle.load(open(inputdir+sys.argv[5])))

    # Wv = np.random.random((10,50))
    # Wv[0] = Wv[0]*0
    print Wv[0]
    # rng = np.random.RandomState(3435)
    PF1 = np.asarray(np.random.uniform(low=-1, high=1, size=[101, 5]), dtype='float32')
    padPF1 = np.zeros((1, 5))
    PF1 = np.vstack((padPF1, PF1))
    PF2 = np.asarray(np.random.uniform(low=-1, high=1, size=[101, 5]), dtype='float32')
    padPF2 = np.zeros((1, 5))
    PF2 = np.vstack((padPF2, PF2))
    print PF1[0]
    print PF2[0]
    trainModel(train,
                    test,
                    dev,
                    50,
    				resultdir,
                    Wv,
                    PF1,
                    PF2,batch=50, test_epoch=0, to_train=1, num_classes=53)
