
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow_probability import distributions as tf_prob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns 

tf.random.set_seed(42)
np.random.seed(42)


class MDN(tf.keras.Model):

    def __init__(self, no_neurons=100, no_components = 2):
        super(MDN, self).__init__(name="MDN")
        self.no_neurons = no_neurons
        self.no_components = no_components
        
        self.hl_1 = Dense(no_neurons, activation="relu", name="hl_1")
        self.hl_2 = Dense(no_neurons, activation="relu", name="hl_2")
        
        self.alpha = Dense(no_components, activation="softmax", name="alpha")
        self.mu = Dense(no_components, activation="nnelu", name="mu")
        self.sigma = Dense(no_components, activation="nnelu", name="sigma")
        self.conc = Concatenate(name="conc")
        
    def call(self, inputs):
        x = self.hl_1(inputs)
        x = self.hl_2(x)
        
        alpha_v = self.alpha(x)
        mu_v = self.mu(x)
        sigma_v = self.sigma(x)
        
        return self.conc([alpha_v, mu_v, sigma_v])
    

def nnelu(input):
    #Calculate non-negative ELU
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))

def vector_unfold(parameter_vector):
    #Unfold the list of given parameters
    return [parameter_vector[:,i*no_components:(i+1)*no_components] for i in range(no_parameters)]

def gnll_loss(y, parameter_vector):
    #Calculate negative log-likelihood
    alpha, mu, sigma = vector_unfold(parameter_vector)
    
    gm = tf_prob.MixtureSameFamily(mixture_distribution=tf_prob.Categorical(probs = alpha),
                               components_distribution=tf_prob.Normal(loc = mu, scale = sigma))
    
    log_likelihood = gm.log_prob(tf.transpose(y))
    
    return -tf.reduce_mean(log_likelihood, axis=-1)

#Add activation function to keras
tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})


#Load data from https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set
df = pd.read_excel('real_estate.xlsx')


#Split in train/test
train = df.iloc[:300,]
test = df.iloc[300:,]

x_train = train.drop(['No','X1 transaction date','Y house price of unit area'], axis = 1)
y_train = train['Y house price of unit area']

x_test = test.drop(['No','X1 transaction date','Y house price of unit area'], axis = 1)
y_test = test['Y house price of unit area']

#Scale values
min_max_scaler = MinMaxScaler()

x_train = min_max_scaler.fit_transform(x_train) 
x_test = min_max_scaler.transform(x_test) 

#Create a simple Feed-Forward Neural Network
model = Sequential()
model.add(Dense(24, input_dim=x_train.shape[1],
                kernel_initializer='normal',
                activation="relu"))
model.add(Dense(1, activation = 'linear'))
model.compile(loss="MeanSquaredError", optimizer='adam')

history = model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose = 0)

#predict on test set
pred = model.predict(x_test)

#Save to dataframe and compare with real values
results_df = pd.DataFrame({'Real':y_test.values, 'Predicted':[w[0] for w in pred]})
results_df.head()

results_df.iloc[18]

#Select parameters
no_parameters = 3 #Should always be 3: alpha, mu and sigma
no_components = 3 #Choose and tune according to the application
no_neurons = 100 #Hyper-parameter to be tuned

#Compile and train MDN
mdn = MDN(no_neurons=no_neurons, no_components=no_components)
mdn.compile(loss=gnll_loss, optimizer='adam')

mdn.fit(x=x_train, y=y_train,epochs=150, validation_data=(x_test, y_test), batch_size=32, verbose=0)

#create plotting function
def plot_rate(ax, index, color_index):
    alpha, mu, sigma = vector_unfold(mdn.predict(x_test[index].reshape(1,-1)))

    gm = tf_prob.MixtureSameFamily(
            mixture_distribution=tf_prob.Categorical(probs=alpha),
            components_distribution=tf_prob.Normal(loc=mu, scale=sigma))
    pyx = gm.prob(x)
    
    ax.plot(x,pyx,alpha=1, color=sns.color_palette()[color_index], linewidth=2, label="PDF for prediction {}".format(index))
    

#Select x values according to the dataset
x = np.linspace(0,100,int(1e3))

plt.figure(figsize=(10,7))
ax = plt.gca()

#Select specific examples to plot
plot_rate(ax, 18,0)
plot_rate(ax, 55,1)

ax.set_xlabel("Price per Unit Area")
ax.set_ylabel("p(Price)")

ax.legend(fontsize=16)
plt.title('Mixture Density Network')
plt.tight_layout()
plt.savefig('pdf_real_estate.jpg')
plt.show()

print('Real Value of observation 18: {}'.format(y_test.iloc[18]))
print('Real Value of observation 55: {}'.format(y_test.iloc[55]))

x = np.linspace(0,100,int(1e3)) 

def find_pdf(index):
    alpha, mu, sigma = vector_unfold(mdn.predict(x_test[index].reshape(1,-1)))

    gm = tf_prob.MixtureSameFamily(
            mixture_distribution=tf_prob.Categorical(probs=alpha),
            components_distribution=tf_prob.Normal(
                loc=mu,       
                scale=sigma))
    pyx = gm.prob(x)
    return pyx

all_preds = np.array([find_pdf(0)])
for i in range(1,x_test.shape[0]):
    temp_pred = np.array([find_pdf(i)])
    all_preds = np.concatenate((all_preds, temp_pred), axis=0)

df_all_preds = pd.DataFrame(all_preds.T)
df_all_preds.index = np.round(x,2)
df_all_preds = df_all_preds.sort_index(ascending = False)

#Plot heatmap
plt.figure(figsize=(12,6))
sns.heatmap(df_all_preds)
plt.savefig('heatmap.jpg')
plt.show()

