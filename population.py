import pydeep.rbm.model as model
import pydeep.rbm.trainer as trainer

import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from matplotlib import pyplot as plt
from scipy.fft import fftshift
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cityblock
import math
import time
import copy
import pickle
import os.path

""" global parameters """
update_offsets = 0.01

#preprocessing data: getting messages of speaker to listener so it can become speaker
batch_size = 14

# Input and hidden dimensionality
v1 = 8
v2 = 7

#concepts and suffixes/prefixes in 1 array
#--> persoonsvorm (0000000): length 7 & suffix? (0 or 1) ; 00000 kmntr ; 000000 ken ko na te ke ka : length 7 & suffix? (0 or 1)
data = np.loadtxt('./training.data')
dataRedundant = np.loadtxt('./trainingRedundant.data')

#concepts with the sinal part put to zero
concepts = np.loadtxt('./speaker_concepts.data')
conceptsRedundant = np.loadtxt('./speaker_conceptsRedundant.data')

#signals with the concept part put to zero
signals = np.loadtxt('./differentSignals.data')
signalsRedundant = np.loadtxt('./differentSignalsRedundant.data')

# Training paramters
batch_size = 14


def rbmTrainer(rbm,kindOfTrainer):
    if kindOfTrainer == 'pcd':
        trainer_ = trainer.PCD(rbm, num_chains=batch_size)
    elif kindOfTrainer == 'cd':
        trainer_ = trainer.CD(rbm)
    elif kindOfTrainer == 'pt':
        trainer_ = trainer.PT(rbm)
    elif kindOfTrainer == 'ipt':
        trainer_ = trainer.IPT(rbm, num_samples=batch_size)
    elif kindOfTrainer == 'gd':
        trainer_ = trainer.GD(rbm)
    return trainer_
""" Training an rbm with the data --> copied from pydeep documentation"""
def train_first_lang(d,hidden_nodes,epochs, kindOfTrainer, eps,centered):

    # Create centered or normal model
    if centered:
        rbm = model.BinaryBinaryRBM(number_visibles=v1 + v2,
                                number_hiddens=hidden_nodes,
                                data=None,
                                initial_weights=0.01,
                                initial_visible_bias='AUTO',
                                initial_hidden_bias='AUTO',
                                initial_visible_offsets='AUTO',
                                initial_hidden_offsets='AUTO')
    else:
        rbm = model.BinaryBinaryRBM(number_visibles= v1 + v2,
                                number_hiddens=hidden_nodes,
                                data=None,
                                initial_weights=0.01,
                                initial_visible_bias=0.0,
                                initial_hidden_bias=0.0,
                                initial_visible_offsets=0.0,
                                initial_hidden_offsets=0.0)

    
    trainer_=rbmTrainer(rbm,kindOfTrainer)

    trainer_.train(data=d,
                            epsilon=eps,
                            num_epochs=epochs,
                            update_visible_offsets=update_offsets,
                            update_hidden_offsets=update_offsets)
       
    return rbm

class FirstLanguageAgent(Agent):
    """An agent with its own rbm. (trained on initialization)"""
    def __init__(self, unique_id, model, rbm,trainer,cpts,mh,epsilonInteractie):
        super().__init__(unique_id, model)
        self.rbm=rbm
        self.trainer = rbmTrainer(rbm,trainer)
        self.epsilon = epsilonInteractie
        self.correctCommunication = 0
        self.cpts = cpts
        self.mh = mh
    #what happens when this agent is assigned to speak
    def getCorrectCommunication(self):
        return self.correctCommunication
    def resetCorrectCommunication(self):
        self.correctCommunication = 0
    
    def step(self):
        #1 find person to talk to (has to be other person so filter)
        people = self.model.schedule.agents
        filtered_people = filter(lambda x: x.unique_id != self.unique_id, people)
        other_agent = self.random.choice(list(filtered_people))
        #2 choose random concept (first 8 nodes of visible nodes) --> draw from distribution??
        rnd_idx = np.random.randint(len(self.cpts)-1)
        concept = self.cpts[rnd_idx]
        #3 get the message for this concept (the last 7 nodes of the visible nodes) --> ?? did not immediately found 1 easy function for this
        msg = self.rbm.sample_v(self.rbm.probability_v_given_h(self.rbm.sample_h(self.rbm.probability_h_given_v(concept))))
        msgreal = msg[0][v1:]
        msgreal = np.concatenate([np.zeros(v1),msgreal])
        #4 give the listener the message, and get its concept for this message as output
        listener_concept = other_agent.listen(msgreal)
        #5 give feedback on the output of the listener
        #5.1 if correct --> communicative succes: count increase and do nothing?? or train??
        if self.mh:
                length = cityblock(listener_concept,concept[:v1])
                #print(str(listener_concept) + " list and og "+str(concept[:v1])+" mhdist: "+str(length))
                #(1-(length/(v1+v2))) --> if want to present the same as correctCommunication
                self.correctCommunication += length
                if (listener_concept != concept[:v1]).all():
                    trainData = np.concatenate([concept[:v1],msg[0][v1:]])
                    other_agent.train(trainData)
        else:
            if (listener_concept == concept[:v1]).all():
            #print("communicative success")
                self.correctCommunication += 1
            else:
            ##5.2 if incorrect --> train?? or do nothing??
                #to test the code, uncomment and see if it always gives 1
                #self.correctCommunication += 1
                trainData = np.concatenate([concept[:v1],msg[0][v1:]])
                other_agent.train(trainData)
                #print("speak:og concept en correct msg "+str(data[rnd_idx])+" message of speaker"+ str(msg[0]) + "  concept listener  " + str(listener_concept))
    
    #when another agent talks to you --> put message in rbm and output your concept for it
    def listen(self, msg):
        output= self.rbm.sample_v(self.rbm.probability_v_given_h(self.rbm.sample_h(self.rbm.probability_h_given_v(msg))))
        return output[0][:v1]
        #updaten (?? zoals bij trainnen)
    
    #when this agent needs to be trained
    def train(self, data):
        #need the same number of data as the batch_size 
        multiple_concept=np.tile(data, (batch_size, 1))
        self.trainer.train(data=multiple_concept,
                #???nog veranderen voor update offsets mss
                            epsilon=self.epsilon,
                            update_visible_offsets=update_offsets,
                            update_hidden_offsets=update_offsets)
    def complexity(self):
        lstSignals=list()
        for concept in self.cpts:
            msg = self.rbm.sample_v(self.rbm.probability_v_given_h(self.rbm.sample_h(self.rbm.probability_h_given_v(concept))))
            msgreal = msg[0][v1:]
            lstSignals.append(msgreal)
        return lstSignals
class AloreseModel(Model):
    """A model with some number of agents."""
    def __init__(self, N,di,d,cpts,mh,hn, epochs,kindOfTrainer,epsilon,centered):
        self.num_agents = N
        #so the agents speak in random order
        self.schedule = RandomActivation(self)
        # Create agents
        filename = "../differentRBMs/%s/hn%depo%dTR%seps%dcentered%s" %(di,hn,epochs,kindOfTrainer,epsilon*100,centered)
        if not(os.path.exists(filename)):
            fileRBM= train_first_lang(d,hn,epochs, kindOfTrainer, epsilon,centered)
            with open(filename, 'wb') as r:
                pickle.dump(fileRBM,r)
            print("new")
        else:
            with open(filename,'rb') as r:
                fileRBM= pickle.load(r)
        for i in range(self.num_agents):
            copiedRBM= copy.deepcopy(fileRBM)
            a = FirstLanguageAgent(i, self, copiedRBM,kindOfTrainer,cpts,mh,epsilon)
            self.schedule.add(a)

    def step(self):
        '''Advance the model by one step --> let each agents speak to another random agent'''
        self.schedule.step()
    def getAgents(self):
        return self.schedule.agents
    def totalCorrect(self):
        total=0
        for i in self.getAgents():
            total += i.getCorrectCommunication()
        return total
    def resetCorrect(self):
        for i in self.getAgents():
            i.resetCorrectCommunication()
    def complexityLists(self,s):
        lstComplexity = list()
        idx=0
        for i in self.getAgents():
            lst = i.complexity()
            if idx==0:
                lstComplexity = lst
                idx=1
            else:
                lstComplexity = np.concatenate([lstComplexity,lst])
        #print(lstComplexity)
        df = pd.DataFrame(columns=('array', 'count'))
        dataframeIdx = 0
        for x in s:
            c = (lstComplexity == x).all(-1).sum()
            df.loc[dataframeIdx] = [x, c]
            dataframeIdx +=1
        return df


def experiment(iterations, model):
    for i in range(iterations):
        model.step()
    return model.totalCorrect()
def experimentIntermediate(iterations,model):
    for i in range(iterations):
        model.step()
    res =model.totalCorrect()
    model.resetCorrect()
    return res
def experimentComplexity(iterations,model,s):
    complexityBefore =model.complexityLists(s)
    for i in range(iterations):
        model.step()
    complexityAfter =model.complexityLists(s)
    correct = model.totalCorrect()
    return complexityBefore,complexityAfter,correct


def getData(redundant,manhatten):
    valueName = 'communicative success'
    if redundant:
        if manhatten:
            di = "savedDataRedundantMH"
            valueName = 'manhattan distance'
        else:
            di = "savedDataRedundant"
        d=dataRedundant
        c=conceptsRedundant
        s=signalsRedundant
    else:
        di = "savedData"
        d=data
        c=concepts
        s=signals
    return valueName,di,d,c,s

#1 columnname gewoon param???
def experimentSingleParameter(param,hn,epochs,kindOfTrainer,epsilon,centered,r,mh,itrRemakeAgents, itrInteractions, nrPeople,plot,dictionary,dataframeIdx=0):
    valueName,di,d,c,s=getData(r,mh)
    start = time.time()
    fName = "people%dinteracts%dremakeA%d" %(nrPeople,itrInteractions,itrRemakeAgents)
    if param=='number of hidden units':
        extra = "HN-epo%dtr-%seps%fcentered%s"%(epochs,kindOfTrainer,epsilon,centered)
        columnName = 'number of hidden units'
        rangePar = hn
        title = 'number of hidden units versus %s' %(valueName)
    elif param=='number of epochs pretraining':
        extra = "hn%dEPO-tr-%seps%fcentered%s"%(hn, kindOfTrainer,epsilon,centered)
        columnName = 'number of epochs pretraining'
        rangePar = epochs
        title = 'number of epochs pretraining versus %s' %(valueName)
    elif param=='trainers':
        extra = "hn%depo%dTR-eps%fcentered%s"%(hn, epochs,epsilon,centered)
        columnName = 'trainers'
        rangePar = kindOfTrainer
        title = 'trainers versus %s' %(valueName)
    elif param=='epsilon':
        extra = "hn%depo%dtr-%sEPScentered%s"%(hn,epochs,kindOfTrainer,centered)
        columnName = 'epsilon'
        rangePar = epsilon
        title = "epsilon for %s versus %s" %(kindOfTrainer,valueName)
    elif param=='centered':
        extra = "hn%depo%dtr-%seps%fCENTERED"%(hn,epochs,kindOfTrainer,epsilon)
        columnName = 'centered'
        rangePar = centered
        title = 'comparing centered and normal RBM with %s' %(valueName)

    df = pd.DataFrame(columns=(columnName, valueName))
    for i in rangePar:
        for j in range(itrRemakeAgents):
            if param=='number of hidden units':
                model = AloreseModel(nrPeople,di,d,c,mh,i, epochs,kindOfTrainer,epsilon,centered)
                par="%d" %(i)
            elif param=='number of epochs pretraining':
                model = AloreseModel(nrPeople,di,d,c,mh,hn, i,kindOfTrainer,epsilon,centered)
                par="%d" %(i)
            elif param=='trainers':
                model = AloreseModel(nrPeople,di,d,c,mh,hn, epochs,i,epsilon,centered)
                par="%s" %(i)
            elif param=='epsilon':
                model = AloreseModel(nrPeople,di,d,c,mh,hn, epochs,kindOfTrainer,i,centered)
                par="%f" %(i)
            elif param=='centered':
                model = AloreseModel(nrPeople,di,d,c,mh,hn, epochs,kindOfTrainer,epsilon,i)
                par="%s" %(i)
            #success = experiment(itrInteractions,model)
            complexityBefore,complexityAfter,success = experimentComplexity(itrInteractions,model,s)
            complexityBefore.to_excel("../%s/%s/complexityBefore/par%s%s%sRemake%d.xlsx" %(di,dictionary,par,extra,fName,j))
            complexityAfter.to_excel("../%s/%s/complexityAfter/par%s%s%sRemake%d.xlsx" %(di,dictionary,par,extra,fName,j))
            success = success/nrPeople/itrInteractions
            df.loc[dataframeIdx] = [i, success]
            dataframeIdx +=1
             #average communicative succes for 1 agent, on a scale from 0 to 1
       #average communicative success for 1 model
            print(str(i))
            print(str(j))
            print(str(success))
        end = time.time()
        print("time elapsed: " + str(end - start))
    if plot:
        filename = "../%s/%s/%s%s.xlsx" %(di,dictionary,extra,fName)
        df.to_excel(filename)
        if param=='kindOfTrainer' or param=='centered':
            sns.barplot(data=df,x=columnName, y=valueName)
        else:
            sns.lineplot(data=df,x=columnName, y=valueName,err_style="bars")
        
        plt.title(title)
        if not(mh):
            plt.ylim([0, 1])
        plt.show()
    else:
        return df

def getNames(redundant,manhatten):
    valueName = 'communicative success'
    if redundant:
        if manhatten:
            di = "savedDataRedundantMH"
            valueName = 'manhattan distance'
        else:
            di = "savedDataRedundant"
    else:
        di = "savedData"
    return valueName,di

def experimentCombination(dictionary,param1,param2,hn,epochs,kindOfTrainer,epsilon,centered,r,mh,itrRemakeAgents, itrInteractions, nrPeople):
    valueName,di = getNames(r,mh)
    #list_gd = epsilonTrainer('gd')
    if param1=='number of hidden units':
        rangePar = hn
    elif param1=='number of epochs pretraining':
        rangePar = epochs
    elif param1=='trainers':
        rangePar = kindOfTrainer
    elif param1=='epsilon':
        rangePar = epsilon
    elif param1=='centered':
        rangePar = centered
    lst=['number of hidden units','number of epochs pretraining','trainers','epsilon','centered']
    lst.remove(param1)
    lst.remove(param2)
    lstTitle=list()
    for el in lst:
        if el == 'number of hidden units':
            s = "hn%d" %(hn)
            lstTitle.append(s)
        if el == 'number of epochs pretraining':
            s = "epo%d" %(epochs)
            lstTitle.append(s)
        if el == 'trainers':
            s = "tr-%s" %(kindOfTrainer)
            lstTitle.append(s)
        if el == 'epsilon':
            s = "eps%f" %(epsilon)
            lstTitle.append(s)
        if el == 'centered':
            s = "centered%s" %(centered)
            lstTitle.append(s)
    frames = list()
    dataframeIdx = 0
    for e in [hn,epochs,kindOfTrainer,epsilon,centered]:
            if isinstance(e, list) and e!=rangePar:
                length = len(e)
    for i in rangePar:
        if param1=='number of hidden units':
            df = experimentSingleParameter(param2,i,epochs,kindOfTrainer,epsilon,centered,r,mh,itrRemakeAgents, itrInteractions,nrPeople,False, dictionary,dataframeIdx)
        elif param1=='number of epochs pretraining':
            df = experimentSingleParameter(param2,hn,i,kindOfTrainer,epsilon,centered,r,mh,itrRemakeAgents, itrInteractions,nrPeople,False, dictionary,dataframeIdx)
        elif param1=='trainers':
            df = experimentSingleParameter(param2,hn,epochs,i,epsilon,centered,r,mh,itrRemakeAgents, itrInteractions,nrPeople,False, dictionary,dataframeIdx)
        elif param1=='epsilon':
            df = experimentSingleParameter(param2,hn,epochs,kindOfTrainer,i,centered,r,mh,itrRemakeAgents, itrInteractions,nrPeople,False, dictionary,dataframeIdx)
        elif param1=='centered':
            df = experimentSingleParameter(param2,hn,epochs,kindOfTrainer,epsilon,i,r,mh,itrRemakeAgents, itrInteractions,nrPeople,False, dictionary,dataframeIdx)
        #rangehn
        
        column = [i for j in range(length*itrRemakeAgents)]
        df[param1] = column
        frames.append(df)
        dataframeIdx += length*itrRemakeAgents
    #plt.plot (list_gd, label= 'gd')
    result = pd.concat(frames)
    filename = "../%s/%s/%s%s%speople%dinteracts%dremakeA%d.xlsx" %(di,dictionary,lstTitle[0],lstTitle[1],lstTitle[2],nrPeople,itrInteractions, itrRemakeAgents)
    result.to_excel(filename)
    sns.lineplot(data=result,x=param1, y=valueName,hue=param2,err_style="bars")
    #sns.barplot(data=result,x=param1, y=valueName,hue=param2)
    title = 'different %s for different %s \n versus %s' %(param1,param2,valueName)
    plt.title(title)
    if not(mh):
        plt.ylim([0, 1])
    plt.show()

def experimentTimesteps(parts,hn,epochs,kindOfTrainer,epsilon,centered,r,mh,itrRemakeAgents, itrInteractions, nrPeople):
    start = time.time()
    valueName,di,d,c,s=getData(r,mh)
    fName = "hn%depo%dtr-%seps%fcentered%speople%dinteracts%dremakeA%d" %(hn,epochs,kindOfTrainer,epsilon,centered,nrPeople,itrInteractions,itrRemakeAgents)
    #period of time
    df = pd.DataFrame(columns=('time interval', valueName))
    dataframeIdx = 0
    
    for j in range(itrRemakeAgents):
        iterPerPart = math.floor(itrInteractions/parts)
        #don't reset count so need to devide between more
        devideIter = 0
        model = AloreseModel(nrPeople,di,d,c,mh,hn, epochs,kindOfTrainer,epsilon,centered)
        for i in range(parts):
            if i == (parts -1):
                iterPerPart += (itrInteractions%parts)
            devideIter += iterPerPart
        
            #print("ok" + str(i) + "ok" +str(j))
            #success = experiment(iterPerPart,model)
            complexityBefore,complexityAfter,success = experimentComplexity(iterPerPart,model,s)
            complexityBefore.to_excel("../%s/timesteps/complexityBefore/interval%d%sRemake%d.xlsx" %(di,i,fName,j))
            complexityAfter.to_excel("../%s/timesteps/complexityAfter/interval%d%sRemake%d.xlsx" %(di,i,fName,j))
            success = success/nrPeople/devideIter
            df.loc[dataframeIdx] = [i, success]
            dataframeIdx +=1
             #average communicative succes for 1 agent, on a scale from 0 to 1
       #average communicative success for 1 model
            print(str(i))
            print(str(j))
            print(str(success))
        end = time.time()
        print("time elapsed: " + str(end - start))
    filename = "../%s/timesteps/%s.xlsx" %(di,fName)
    df.to_excel(filename)
    sns.lineplot(data=df,x="time interval", y=valueName, err_style="bars")
    title = '%s of the intermediate interactions' %(valueName)
    plt.title(title)
    if not(mh):
        plt.ylim([0, 1])
    plt.show()

tepochs = 100000
thn = 14
tcentered = True
#???nog weg doen

tkindOfTrainer = 'cd'
tepsilon = 0.98
#tepsilon = 0.05
nrPeople =3
#??comment
"""
v1 = 16
v2 = 14
redundant = True
mh = True"""

redundant = False
mh = False
#savedData
experimentSingleParameter('number of hidden units',range(10,25),tepochs,tkindOfTrainer,tepsilon,tcentered,redundant,mh, 3, 1000,100,True, 'hntest')

#[100,500,1000,5000,10000,50000,100000]
#experimentSingleParameter('number of epochs pretraining',thn,[100,500,1000,5000,10000],tkindOfTrainer,tepsilon,tcentered,redundant,mh, 3, 100,10,True, 'epochs')

#kindOfTrainer
#experimentSingleParameter('trainers',thn,tepochs,['cd','pcd','pt','ipt'],tepsilon,tcentered,redundant,mh, 3, 1000,100,True, 'kindOfTrainer')

#epsilon
#experimentSingleParameter('epsilon',thn,tepochs,tkindOfTrainer,[0.01,0.02,0.05,0.07,0.1,0.5,0.7,0.9],tcentered,redundant,mh, 3, 1000,100,True, 'epsilon')

#centered
#experimentSingleParameter('centered',thn,tepochs,tkindOfTrainer,tepsilon,[True,False],redundant,mh, 3, 1000,100,True, 'centered')


#timesteps
#different combos
#epoVsHn
#epsVsHn
#epsVsTr
#hnVsTr
#...
#kijk of barplot nodig of niet! --> in alle dictionaries aanpassen, ook MH

#experimentCombination('epsVsTr','epsilon','trainers',thn,tepochs,['cd','pcd','pt'],[0.01,0.05,0.1,0.5,0.9],tcentered,redundant,False,2, 100, 10)

#experimentCombination('epoVsHn','number of hidden units','number of epochs pretraining',range(10,30,10),[100,500,1000],tkindOfTrainer,tepsilon,tcentered,redundant,False,2, 100, 10)

#experimentTimesteps(10,thn,tepochs,tkindOfTrainer,tepsilon,tcentered,redundant,mh, 3, 1000,100)