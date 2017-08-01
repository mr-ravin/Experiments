import tensorflow as tf
import numpy as np
n_nodes_hl1=4 #500
n_nodes_hl2=4 #500
n_nodes_hl3=4 #500

label=[]
input_data=[]
fin_input=[]
file_data=open("Skin_NonSkin.txt")
data_read=file_data.readline()
while data_read!="":
 data_read=data_read.split("\t")
 data_read[3]=data_read[3][:-1]
 ty=int(data_read[-1])-1
 if ty==0:
   label.append([1,0])
 if ty==1:
   label.append([0,1])
 tmp=data_read[:-1]
 for i in range(len(tmp)):
   tmp[i]=int(tmp[i])
 input_data.append(tmp)
 data_read=file_data.readline()

print("len data = ",len(input_data))
train_x=input_data[:1000]+input_data[-1000:]+input_data[1000:2000]+input_data[-2000:-1000]
train_y=label[:1000]+label[-1000:]+label[1000:2000]+label[-2000:-1000]
n_classes=2
batch_size=1

x=tf.placeholder(tf.float32,[None,3])
y=tf.placeholder(tf.float32,[None,2])


############
#This function swaps the weights w.r.t the column of weight matrix.
def swap_column(arr):
  size=arr.shape
  row=size[0]
  colmn=size[1]
  res=arr[:]
  for i in range(colmn):
    tmp=[]
    
    for j in range(row): 
      tmp.append(arr[j][i]) 
    tmp1=sorted(tmp)
    
    for it in range(len(tmp)):
      cnt=0
      for jt in range(len(tmp)):
        if it != jt:
          if tmp[it] >=tmp[jt]:
            cnt=cnt+1
      res[it][i]=tmp1[len(tmp1)-1-cnt] 
  return res


###########
#This function swaps the weights w.r.t the row of weight matrix.
def swap_row(arr):
  size=arr.shape
  row=size[0]
  colmn=size[1]
  res=arr[:]
  for i in range(row): 
    tmp=[]
    
    for j in range(colmn): 
      tmp.append(arr[i][j]) 
    tmp1=sorted(tmp)
    
    for it in range(len(tmp)):
      cnt=0
      for jt in range(len(tmp)):
        if it != jt:
          if tmp[it] >=tmp[jt]:
            cnt=cnt+1
      res[i][it]=tmp1[len(tmp1)-1-cnt] 
  return res

###########
#This function swaps the weights w.r.t the individual layers.
def swap(arr):
  size=arr.shape
  row=size[0]
  colmn=size[1]
  rest=arr[:]
  res_tmp=arr[:]
  res_t=[]

  for t in arr:
    for u in t:
      res_t.append(u)

  rtm=sorted(res_t)
  rtm_cpy=[]
  for tyu in range(len(rtm)):
    rtm_cpy.append(0)
  print(rtm)

  for it in range(len(res_t)):
    cnt=0
    for jt in range(len(res_t)):
      if it!=jt:
        if res_t[it]>res_t[jt]:
          cnt=cnt+1
    rtm_cpy[it]=rtm[len(res_t)-1-cnt]

  count_num=0
  rw=0
  for wd in rtm_cpy:
    #print(wd)
    if count_num<colmn:
      res_tmp[rw][count_num]=wd
      count_num=count_num+1
    else:
      count_num=0
      rw=rw+1
      res_tmp[rw][count_num]=wd
      count_num=count_num+1
  return res_tmp

#############





def neural_network_node(data):
  hidden_1_layer={'weights':tf.Variable(tf.random_normal([3,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
  output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
  l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
  l1=tf.nn.relu(l1)  
  output=tf.add(tf.matmul(l1,output_layer['weights']),output_layer['biases'])

  return [output,hidden_1_layer['weights'],hidden_1_layer['biases'],output_layer['weights'],output_layer['biases']]



def train_neural_network(x):
  prediction_t=neural_network_node(x)
  prediction=prediction_t[0]
  predict=tf.nn.softmax(prediction)
  wt1=prediction_t[1]
  b1=prediction_t[2]
  wt2=prediction_t[3]
  b2=prediction_t[4]
  
  tdy=tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)
  cost=tf.reduce_mean(tdy)
  optimizer=tf.train.AdamOptimizer().minimize(cost)
  
  hm_epochs=10
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(hm_epochs):
      epoch_loss=0
      i=0
      while i<len(train_x):
        start=i
        end=i+batch_size
        epoch_x=np.array(train_x[start:end])
        epoch_y=np.array(train_y[start:end])
          
        _,pred,c,wtr1,br1,wtr2,br2=sess.run([optimizer,predict,cost,wt1,b1,wt2,b2],feed_dict={x:epoch_x,y:epoch_y})
        print("prediction =",pred)
        epoch_loss=epoch_loss+c
        i=i+batch_size
      
    correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct,'float'))
    return [wtr1,br1,wtr2,br2]
    

tpl=train_neural_network(x)

tpl[0]=np.array(tpl[0])
tpl[1]=np.array(tpl[1])
tpl[2]=np.array(tpl[2])
tpl[3]=np.array(tpl[3])
print("weights layer 1 :")
print(tpl[0])
print("biases layer 1 :")
print(tpl[1])
print("weights layer 2 :")
print(tpl[2])
print("biases layer 2 :")
print(tpl[3])


n_nodes_hl1=4 #500
n_nodes_hl2=4 #500
n_nodes_hl3=4 #500

def neural_network_node2(data):
  hidden_1_layer={'weights':tpl[0],'biases':tpl[1]}
  #hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1 , n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}  
  #hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2 , n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
  output_layer={'weights':tpl[2],'biases':tpl[3]}
  
  l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
  l1=tf.nn.relu(l1)

  output=tf.add(tf.matmul(l1,output_layer['weights']),output_layer['biases'])
  return output
  

def predict_neural_network(x):
  prediction=tf.nn.softmax(neural_network_node2(x))
  #print("prediction === ",prediction)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pred=sess.run(prediction,{x:train_x})
    return pred


pred1=predict_neural_network(x)

############swapping

tpl[0]=swap(tpl[0])   #swap_row(tpl[0])   #swap_column(tpl[0])

tpl[2]=swap(tpl[2])   #swap_row(tpl[2])   #swap_column(tpl[2])

############swapping

print("new weights layer 1 :")
print(tpl[0])
print("new biases layer 1 :")
print(tpl[1])
print("new weights layer 2 :")
print(tpl[2])
print("new biases layer 2 :")
print(tpl[3])

############

pred2=predict_neural_network(x)
print("pred 1 :\n",pred1)
print("pred 2 :\n",pred2)
print("fin===\n",pred1-pred2)

