import pydeep.rbm.model as model
import pydeep.rbm.trainer as trainer

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cityblock
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

""" Training an rbm with the data --> copied from pydeep documentation"""
def kindOfRbm(centered,hidden_nodes):
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
    return rbm
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
    rbm = kindOfRbm(centered, hidden_nodes)
    trainer_=rbmTrainer(rbm,kindOfTrainer)
    for epoch in range(1, epochs + 1):

            # Loop over all batches
        batch = d
        trainer_.train(data=batch,
                            epsilon=eps,
                            update_visible_offsets=update_offsets,
                            update_hidden_offsets=update_offsets)

            # Calculate reconstruction error and expected end time every 10th epoch
        """if epoch % 1000000 == 0:
            RE = np.mean(estimator.reconstruction_error(rbm, d))
            print('{}\t\t{:.4f}'.format(
                epoch, RE, ))"""
    return rbm

#?? add secondLanguageAgent with other data given to train function probably
class FirstLanguageAgent(Agent):
    """An agent with its own rbm. (trained on initialization)"""
    
    def __init__(self, unique_id,gen,cpts,hidden_nodes, centered, kindOfTrainer, epsilon):
        # Create centered or normal model
        if not(gen =='initial'):
            self.rbm = kindOfRbm(centered, hidden_nodes)
            self.trainer = rbmTrainer(self.rbm,kindOfTrainer)
        self.hn =hidden_nodes
        self.kindOfTrainer = kindOfTrainer
        self.epsilon = epsilon
        self.centered =centered
        self.cpts = cpts
        #self.correctCommunication = 0
    #what happens when this agent is assigned to speak
    #def getCorrectCommunication(self):
     #   return self.correctCommunication
    #def resetCorrectCommunication(self):
     #   self.correctCommunication = 0
    def initialSpeaker(self,epochs,d,di):
        if di == 'IL':
            di='savedData'
        elif di == 'ILRedundant':
            di='savedDataRedundantMH'
        elif di == 'ILRedundantMH':
            di = 'savedDataRedundantMH'

        filename = "../differentRBMs/%s/hn%depo%dTR%seps%dcentered%s" %(di,self.hn,epochs,self.kindOfTrainer,self.epsilon*100,self.centered)
        if not(os.path.exists(filename)):
            fileRBM= train_first_lang(d,self.hn,epochs, self.kindOfTrainer, self.epsilon,self.centered)
            with open(filename, 'wb') as r:
                pickle.dump(fileRBM,r)
            print("new")
        else:
            with open(filename,'rb') as r:
                fileRBM= pickle.load(r)
        self.rbm = copy.deepcopy(fileRBM)

    
    def speak(self):
        batch = list()
        for i in range(batch_size):
            rnd_idx = np.random.randint(len(self.cpts)-1)
            concept = self.cpts[rnd_idx]
            #3 get the message for this concept (the last 7 nodes of the visible nodes) --> ?? did not immediately found 1 easy function for this
            msg = self.rbm.sample_v(self.rbm.probability_v_given_h(self.rbm.sample_h(self.rbm.probability_h_given_v(concept))))
            d = np.concatenate([concept[:v1],msg[0][v1:]])
            batch.append(d)
        return batch
        
    #when another agent talks to you --> put message in rbm and output your concept for it
    #????also speak and only train when fault or act like teachers
    def listenAndTrain(self, d):
        multiple_concept=np.tile(d, (batch_size, 1))
        self.trainer.train(data=multiple_concept,
                #???nog veranderen voor update offsets mss
                            epsilon=self.epsilon,
                            update_visible_offsets=update_offsets,
                            update_hidden_offsets=update_offsets)
    def listen(self, msg):
        output= self.rbm.sample_v(self.rbm.probability_v_given_h(self.rbm.sample_h(self.rbm.probability_h_given_v(msg))))
        return output[0][:v1]
        #updaten (?? zoals bij trainnen)
    def evaluate(self, agent,mh):
        #2 choose random concept (first 8 nodes of visible nodes) --> draw from distribution??
        rnd_idx = np.random.randint(len(self.cpts)-1)
        concept = self.cpts[rnd_idx]
        #3 get the message for this concept (the last 7 nodes of the visible nodes) --> ?? did not immediately found 1 easy function for this
        msg = self.rbm.sample_v(self.rbm.probability_v_given_h(self.rbm.sample_h(self.rbm.probability_h_given_v(concept))))
        msgreal = msg[0][v1:]
        msgreal = np.concatenate([np.zeros(v1),msgreal])
        #4 give the listener the message, and get its concept for this message as output
        listener_concept = agent.listen(msgreal)
        #printing the interaction
        #print ("agent " + str(self.unique_id) +"talks to" + str(other_agent.unique_id)+ "concept: "+ str(concept[:v1]) + " ;msg:" +str(msgreal) + " ;output_concept" + str(listener_concept))
        #5 give feedback on the output of the listener
        #5.1 if correct --> communicative succes: count increase and do nothing?? or train??
        if mh:
                length = cityblock(listener_concept,concept[:v1])
                #print(str(listener_concept) + " list and og "+str(concept[:v1])+" mhdist: "+str(length))
                #(1-(length/(v1+v2))) --> if want to present the same as correctCommunication
                return length
                
        else:
            if (listener_concept == concept[:v1]).all():
            #print("communicative success")
                return 1
            else:
                return 0   
    def complexity(self,s):
        lstSignals=list()
        for concept in self.cpts:
            msg = self.rbm.sample_v(self.rbm.probability_v_given_h(self.rbm.sample_h(self.rbm.probability_h_given_v(concept))))
            msgreal = msg[0][v1:]
            lstSignals.append(msgreal)
        df = pd.DataFrame(columns=('array', 'count'))
        dataframeIdx = 0
        for x in s:
            c = (lstSignals == x).all(-1).sum()
            df.loc[dataframeIdx] = [x, c]
            dataframeIdx +=1
        return df  

thn = 14
tcentered = True
tkindOfTrainer = 'pt'
tepsilon = 0.7 
tepochs = 500000
def evaluateAverage(evaluator, agent,interactions,mh):
    values = 0
    for i in range(interactions):
        values += evaluator.evaluate(agent,mh)
        #no Nrpeople, one person represents generation
    values = values/interactions
    return values
def experiment(mh,generations, learning,d,di,cpts,hn,epochs, kindOfTrainer,epsilon,centered):
    initialAgent = FirstLanguageAgent(0, 'initial', cpts, hn,centered, kindOfTrainer, epsilon)
    initialAgent.initialSpeaker(epochs,d,di)
    firstAgent = initialAgent
    agent = initialAgent
    for i in range(generations-1):
        agent = FirstLanguageAgent(i+1, 'normal', cpts, hn,centered, kindOfTrainer, epsilon)
        for j in range(learning):
            batch = firstAgent.speak()
            agent.listenAndTrain(batch)
        #otherwise will evaluate with itself and not previous
    return evaluateAverage(initialAgent, agent,1000,mh),evaluateAverage(firstAgent, agent,1000,mh)
def experimentComplexity(s,mh,generations, learning,d,di,cpts,hn,epochs, kindOfTrainer,epsilon,centered):
    initialAgent = FirstLanguageAgent(0,'initial', cpts,hn,centered, kindOfTrainer, epsilon)
    initialAgent.initialSpeaker(epochs,d,di)
    firstAgent = initialAgent
    agent = initialAgent
    for i in range(generations-1):
        agent = FirstLanguageAgent(i+1, 'normal', cpts, hn,centered, kindOfTrainer, epsilon)
        for j in range(learning):
            batch = firstAgent.speak()
            agent.listenAndTrain(batch)
    return evaluateAverage(initialAgent, agent,1000,mh),evaluateAverage(firstAgent, agent,1000,mh),initialAgent.complexity(s),firstAgent.complexity(s)
def experimentTime(interval,s,mh,generations, learning,d,di,cpts,hn,epochs, kindOfTrainer,epsilon,centered):
    initialAgent = FirstLanguageAgent(0, 'initial', cpts, hn,centered, kindOfTrainer, epsilon)
    initialAgent.initialSpeaker(epochs,d,di)
    firstAgent = initialAgent
    agent = initialAgent
    evaluateInitialList = list()
    evaluateList = list()
    complexList = [initialAgent.complexity(s)]
    evaluate = interval
    for i in range(generations-1):
        agent = FirstLanguageAgent(i+1,'normal', cpts, hn,centered, kindOfTrainer, epsilon)
        for j in range(learning):
            batch = firstAgent.speak()
            agent.listenAndTrain(batch)
        
        if evaluate == 0:
            evaluateInitialList.append(evaluateAverage(initialAgent, agent,1000,mh))
            evaluateList.append(evaluateAverage(firstAgent, agent,1000,mh))
            complexList.append(agent.complexity(s))
            evaluate = interval
        else:
            evaluate -=1
    if evaluate < interval:
        evaluateInitialList.append(evaluateAverage(initialAgent, agent,1000,mh))
        evaluateList.append(evaluateAverage(firstAgent, agent,1000,mh))
        complexList.append(agent.complexity(s))
    return evaluateInitialList,evaluateList,complexList

def getData(redundant,manhatten):
    valueName = 'communicative success'
    if redundant:
        if manhatten:
            di = "ILRedundantMH"
            valueName = 'manhattan distance'
        else:
            di = "ILRedundant"
        d=dataRedundant
        c=conceptsRedundant
        s=signalsRedundant
    else:
        di = "IL"
        d=data
        c=concepts
        s=signals
    return valueName,di,d,c,s

#1 columnname gewoon param???
def experimentSingleParameter(param,hn,epochs,kindOfTrainer,epsilon,centered,r,mh,itrRemakeAgents, itrLearning, gen,plot,dictionary,dataframeIdx=0):
    valueName,di,d,c,s=getData(r,mh)
    start = time.time()
    fName = "gen%dinteracts%dremakeA%d" %(gen,itrLearning,itrRemakeAgents)
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

    df = pd.DataFrame(columns=(columnName, valueName, 'agent which evaluates'))
    for i in rangePar:
        for j in range(itrRemakeAgents):
            if param=='number of hidden units':                            
                valueInitial, value,complexityInitial,complexityPrevious = experimentComplexity(s,mh,gen, itrLearning,d,di,c,i,epochs,kindOfTrainer,epsilon,centered)
                #valueInitial, value = experiment(mh,gen, itrLearning,d,c,i,epochs,kindOfTrainer,epsilon,centered)
                par="%d" %(i)
            elif param=='number of epochs pretraining':
                valueInitial, value,complexityInitial,complexityPrevious = experimentComplexity(s,mh,gen, itrLearning,d,di,c,hn,i,kindOfTrainer,epsilon,centered)
                #valueInitial, value = experiment(mh,gen, itrLearning,d,c,hn,i,kindOfTrainer,epsilon,centered)
                par="%d" %(i)
            elif param=='trainers':
                valueInitial, value,complexityInitial,complexityPrevious = experimentComplexity(s,mh,gen, itrLearning,d,di,c,hn,epochs,i,epsilon,centered)
                #valueInitial, value = experiment(mh,gen, itrLearning,d,c,hn,epochs,i,epsilon,centered)
                par="%s" %(i)
            elif param=='epsilon':
                valueInitial, value,complexityInitial,complexityPrevious = experimentComplexity(s,mh,gen, itrLearning,d,di,c,hn,epochs,kindOfTrainer,i,centered)
                #valueInitial, value = experiment(mh,gen, itrLearning,d,c,hn,epochs,kindOfTrainer,i,centered)
                par="%f" %(i)
            elif param=='centered':
                valueInitial, value,complexityInitial,complexityPrevious = experimentComplexity(s,mh,gen, itrLearning,d,di,c,hn,epochs,kindOfTrainer,epsilon,i)
                #valueInitial, value = experiment(mh,gen, itrLearning,d,c,hn,epochs,kindOfTrainer,epsilon,i)
                par="%s" %(i)
            #success = experiment(itrInteractions,model)
            complexityInitial.to_excel("../%s/%s/complexityInitial/par%s%s%sRemake%d.xlsx" %(di,dictionary,par,extra,fName,j))
            complexityPrevious.to_excel("../%s/%s/complexityPrevious/par%s%s%sRemake%d.xlsx" %(di,dictionary,par,extra,fName,j))
            df.loc[dataframeIdx] = [i, valueInitial,'initial']
            df.loc[dataframeIdx+1] = [i, value,'previous']
            dataframeIdx +=2
             #average communicative succes for 1 agent, on a scale from 0 to 1
       #average communicative success for 1 model
            print(str(i))
            print(str(j))
            print(str(value))
        end = time.time()
        print("time elapsed: " + str(end - start))
    if plot:
        filename = "../%s/%s/%s%s.xlsx" %(di,dictionary,extra,fName)
        df.to_excel(filename)
        if param=='kindOfTrainer' or param=='centered':
            sns.barplot(data=df,x=columnName, y=valueName,hue="agent which evaluates")
        else:
            sns.lineplot(data=df,x=columnName, y=valueName,hue="agent which evaluates",err_style="bars")
        
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
            di = "ILRedundantMH"
            valueName = 'manhattan distance'
        else:
            di = "ILRedundant"
    else:
        di = "IL"
    return valueName,di

def experimentCombination(dictionary,param1,param2,hn,epochs,kindOfTrainer,epsilon,centered,r,mh,itrRemakeAgents, itrLearning, gen):
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
            df = experimentSingleParameter(param2,i,epochs,kindOfTrainer,epsilon,centered,r,mh,itrRemakeAgents, itrLearning,gen,False, dictionary,dataframeIdx)
        elif param1=='number of epochs pretraining':
            df = experimentSingleParameter(param2,hn,i,kindOfTrainer,epsilon,centered,r,mh,itrRemakeAgents, itrLearning,gen,False, dictionary,dataframeIdx)
        elif param1=='trainers':
            df = experimentSingleParameter(param2,hn,epochs,i,epsilon,centered,r,mh,itrRemakeAgents, itrLearning,gen,False, dictionary,dataframeIdx)
        elif param1=='epsilon':
            df = experimentSingleParameter(param2,hn,epochs,kindOfTrainer,i,centered,r,mh,itrRemakeAgents, itrLearning,gen,False, dictionary,dataframeIdx)
        elif param1=='centered':
            df = experimentSingleParameter(param2,hn,epochs,kindOfTrainer,epsilon,i,r,mh,itrRemakeAgents, itrLearning,gen,False, dictionary,dataframeIdx)
        #rangehn
        
        column = [i for j in range(length*itrRemakeAgents*2)]
        df[param1] = column
        frames.append(df)
        dataframeIdx += length*itrRemakeAgents
    #plt.plot (list_gd, label= 'gd')
    result = pd.concat(frames)
    filename = "../%s/%s/%s%s%sgen%dinteracts%dremakeA%d.xlsx" %(di,dictionary,lstTitle[0],lstTitle[1],lstTitle[2],gen,itrLearning, itrRemakeAgents)
    result.to_excel(filename)
    sns.catplot(data=result,x=param1, y=valueName,hue=param2,col="agent which evaluates",kind="bar")
    #sns.barplot(data=result,x=param1, y=valueName,hue=param2)
    title = 'different %s for different %s \n versus %s' %(param1,param2,valueName)
    #plt.title(title)
    """if not(mh):
        plt.ylim([0, 1])"""
    plt.show()

def experimentTimesteps(parts,hn,epochs,kindOfTrainer,epsilon,centered,r,mh,itrRemakeAgents, itrLearning,gen):
    start = time.time()
    valueName,di,d,c,s=getData(r,mh)
    fName = "hn%depo%dtr-%seps%fcentered%sgen%dinteracts%dremakeA%d" %(hn,epochs,kindOfTrainer,epsilon,centered,gen,itrLearning,itrRemakeAgents)
    #period of time
    df = pd.DataFrame(columns=('time interval', valueName,'agent which evaluates'))
    dataframeIdx = 0
    
    for j in range(itrRemakeAgents):
        evaluateInitialList,evaluateList,complexList = experimentTime(parts,s, mh,gen, itrLearning,d,di,c,hn,epochs, kindOfTrainer,epsilon,centered)
        for i in range(len(evaluateList)):
            complexList[i].to_excel("../%s/timesteps/complexity/interval%d%sRemake%d.xlsx" %(di,i,fName,j))
            valueInitial = evaluateInitialList[i]
            value = evaluateList[i]
            df.loc[dataframeIdx] = [i+1, valueInitial,'initial']
            df.loc[dataframeIdx+1] = [i+1, value,'previous']
            dataframeIdx +=2
        print(j)
        end = time.time()
        print("time elapsed: " + str(end - start))
    filename = "../%s/timesteps/%s.xlsx" %(di,fName)
    df.to_excel(filename)
    sns.lineplot(data=df,x="time interval", y=valueName, hue="agent which evaluates",err_style="bars")
    title = '%s of the intermediate interactions' %(valueName)
    plt.title(title)
    if not(mh):
        plt.ylim([0, 1])
    plt.show()

tepochs = 1000000
#2times batchsize
thn = 14
tcentered = True
#???nog weg doen

tkindOfTrainer = 'cd'
tepsilon = 0.05  

nrPeople =3
#??comment

v1 = 16
v2 = 14
redundant = True
mh = True
"""
redundant = False
mh = False"""
#savedData
experimentSingleParameter('number of hidden units',range(10,25),tepochs,tkindOfTrainer,tepsilon,tcentered,redundant,mh, 3, 1000,10,True, 'hn')

#[100,500,1000,5000,10000,50000,100000]
#experimentSingleParameter('number of epochs pretraining',thn,[100,500,1000,5000,10000],tkindOfTrainer,tepsilon,tcentered,redundant,mh, 3, 100,10,True, 'epochs')

#kindOfTrainer
#experimentSingleParameter('trainers',thn,tepochs,['cd','pcd','pt','ipt'],tepsilon,tcentered,redundant,mh, 3, 1000,10,True, 'kindOfTrainer')

#epsilon
#experimentSingleParameter('epsilon',thn,tepochs,tkindOfTrainer,[0.01,0.02,0.05,0.07,0.1,0.5,0.7,0.9],tcentered,redundant,mh, 3, 1000,10,True, 'epsilon')

#centered
#experimentSingleParameter('centered',thn,tepochs,tkindOfTrainer,tepsilon,[True,False],redundant,mh, 3, 1000,10,True, 'centered')


#timesteps
#different combos
#epoVsHn
#epsVsHn
#epsVsTr
#hnVsTr
#...
#kijk of barplot nodig of niet! --> in alle dictionaries aanpassen, ook MH

#experimentCombination('epsVsTr','epsilon','trainers',thn,tepochs,['cd','pcd'],[0.01,0.05,0.1],tcentered,redundant,mh,2, 100, 10)

#experimentCombination('epoVsHn','number of hidden units','number of epochs pretraining',range(10,30,10),[100,500,1000],tkindOfTrainer,tepsilon,tcentered,redundant,False,2, 100, 10)

#experimentTimesteps(10,thn,tepochs,tkindOfTrainer,tepsilon,tcentered,redundant,mh, 3, 1000,30)