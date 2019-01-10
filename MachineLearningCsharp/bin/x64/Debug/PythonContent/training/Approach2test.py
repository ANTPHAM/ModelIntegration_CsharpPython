#==============================================================================
# coding: utf-8
    
# A Hybrid Approach Attempt to make recommendation by Product Ranking
    
# The goal is to build a hybrid model combining the outcomes of different learning methods: Content_Based Filtering & Distribution Based Similarity Mesure ( Heuristic Calculation Approach) and a Naive Bayes Probability Classifier ( Model Based Approach).
    
# This model will be able to rank products,so make recommendations, having taken in account multidimensional parameters.
    
#==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import itertools
from collections import Counter
import math
import operator
ageofqueen=50
AGEL = 50
def test_py():
    return [1, "string", False]

def test1_py():
	return {"age":25}




def JOB()    :

    # create a list of products which will be outcomes to predict for the model
    product_list= ['FUMAX','AVECESAR','TOURISTE','TARTARE','HAMPE','SAUMON','NUGGETS','BRIE','MINIMAX',
                   'MAXPOUSSIN','MAXCOCOTTE','MAXHALEINE','MAXINUS','MAXCHAMPETRE','MAXIFLETTE','XAMAX',
    				   'MAXPARTOUT','MAXNAUDOU','PLANTAMAX','GROSSEFRITE','PATATEDOUCE','SALADEASIAT','DUOSALADE',
    				   'TAPAS','GATEAUCAROTTE','CREMEBRULEE','MOUSSECHOCO','CROQUANT','SOUPEFRUIT','CAFEGOURMAND',
    				   'MENU HAMBOURGEOIS','FRAICHEUR','MOUFLET','MAXNAUDOU_SUPMENU','EVIAN','BADOIT 33cl','ABATILLES PLATES',
    				   'ABATILLES RED','KIR','KIR ROYAL','PUNCH Maison','SANGRIA','RHUM ARRANGE','PICHET PUNCH',
    				   'PICHET SANGRIA','JUS FRUIT MAISON','THE GLACE','COCA','COCA ZERO','OASIS','SCHWEPPES','SIROP',
    				   'RED BULL','ORANGINA','WHISKY','RHUM','VODKA','GIN','GET','TEQUILA','MALIBU','BAILEYS','RICARD',
    				   'PASTIS','PINEAU','PORTO','LILLET','MARTINI','CHAMPAGNE BOUTEILLE','CHAMPAGNE COUPE','PELFORTH',
    				   'PINT PELFORTH','GRIMBERGEN','HOEGGARDEN','DESPERADOS','KEKETTE EXTRA','KEKETTE RED',
    				   'BIERE SANS GLUTEN','EXPRESSO','DECA','DOUBLE EXPRESSO','CAPPUCCINO','LATTE MACCHIATTO',
    				   'MOKACCINO','CHOCOLAT','THE INFUSION','PAPOLLE BLANC SEC','VERRE PAPOLLE BLANC SEC',
    				   'PAPOLLE BLANC MOEL','VERRE PAPOLLE BLANC MOEL','PAPOLLE ROSE','VERRE PAPOLLE ROSE',
    				   'PAPOLLE ROUGE','MAX CUVEE','VERRE MAX CUVEE','VERRE PAPOLLE ROUGE','MENU PLAT',
    				   'HAMBOURGEOIS DU MOMENT','MAXINUS','XAMAX','MAXNAUDOU','SO MONAX','POULETMAX',
    				   'FRITE PDT','PATATEDOUCE','SALADE VERTE','AVE CESAR TERRE','MOUSSECHOCO','SOUPEFRUIT',
    				   'COOKIES','MUFFINS','TARTELETTE AUX FRUITS','CROISSANT','MAX BELLE GUEULE','MAX MENU',
    				   'COCA','COCA ZERO','OASIS','ORANGINA','SCHWEPPES AGRUMES','PAGO ORANGE','EVIAN','BADOIT',
    				   'BIERE PRESSION','EXPRESSO','DECA','BIERE PRESSION MAX MENU','MAXINUS SD','XAMAX SD',
    				   'MAXNAUDOU SD','SO MONAX','POULETMAX','FRITE PDT SD','PATATEDOUCE SD','SALADE VERTE SD',
    				   'AVE CESAR TERRE SD','MOUSSECHOCO SD','SOUPEFRUIT SD','COOKIES','MUFFINS',
    				   'TARTELETTE AUX FRUITS','CROISSANT','MAX BELLE GUEULE','MAX MENU SD','COCA SD',
    				   'COCA SD','OASIS SD','ORANGINA SD','SCHWEPPES AGRUMES SD','PAGO ORANGE SD','EVIAN SD',
    				   'BADOIT SD','BIERE PRESSION','EXPRESSO SD','DECA SD','THE SD','INFUSION SD',
    				   'BIERE PRESSION MAX MENU','BIERE PINTH','ABATILLE','MAXCOCOTTE MENU','MAXHALEINE MENU',
    				   'MAXINUS MENU','MAXCHAMPETRE MENU','MAXIFLETTE MENU','XAMAX MENU','MAXPARTOUT MENU',
    				   'MAXNAUDOU MENU','PLANTAMAX MENU','HAMBOURGEOIS DU MOMENT MENU','PLAT DU MOMENT',
    				   'COMPOTEE CERISE NOIRE','SAUCE POIVRE','Crème au bleu','SAUCE BARBECUE','MAXYONNAISE',
    				   'PESTO','CREME POIVRE VERT','CAFEGOURMANDMENU']
    order= np.random.choice(product_list, size=10,replace=True)
    pd.DataFrame(order)
    pd.DataFrame(pd.Series(Counter(product_list))).T
    
    
    	# Creating an artificiel data set : 20 products x 50 different contexts
    np.random.seed(2)
    matrix= pd.DataFrame(np.random.randint(50,size=(50,20)),columns=np.random.choice(product_list, size=20,replace=True))
    ctx=pd.DataFrame(s for s in ['Context_%d' % i for i in range(1,51)])
    matrix.insert(0,"context",ctx)
    
    #==============================================================================
    # PROBABILTY:  Pr(A | B) = Pr(B | A) x Pr(A)/Pr(B)
    # In our example, this formula becomes:
    # Pr(Product A | Context C) = Pr(Context C | Product A) x Pr(Product A) / Pr(Context C)
    # Pr(Product A) is the probability that a randomly selected product will be this product, so it’s just the number of products in the category divided by the total number of products.
    #==============================================================================
    
    #Check of the Distribution of Products by Contexts
    
    #matrix.hist(figsize=[20,15]);
    
    m=matrix.iloc[:,1:]
    m=m.values
    m=m/np.sum(m,axis=0,keepdims=True)
    PrCAmatrix=pd.DataFrame(m,columns=matrix.iloc[:,1:].columns)# to check the sum of "np.sum(m['CAFEGOURMAND'])=1"
    PrCAmatrix.insert(0,"context",ctx)
    
    Pr_A=pd.DataFrame.sum(matrix.iloc[:,1:])
    Pr_A=Pr_A/np.sum(Pr_A)
    Pr_A=pd.Series(Pr_A)
    	
    
    # By default, we assign the same probability to all of 50 contexts in this example, so we obtain:
    Pr_ctx=1/50 
    
    #==============================================================================
    # PREDICTION ON NEW DATA
    #==============================================================================
    
    
    # Building a function that predicts new data
    def prob_ctx(context,product):
        prob= PrCAmatrix.loc[(PrCAmatrix['context']==context),[product]]*Pr_A[product]/Pr_ctx
        return prob.values[0][0]
    print (" The probability that a given user takes a 'CAFEGOURMAND' when he is in the Context_10 is:",
    		   "%.3f" % prob_ctx('Context_10','CAFEGOURMAND') )
    
    print (" The probability that a given user takes a 'CAFEGOURMAND' when he is in the Context_25 is:",
    		   "%.3f" % prob_ctx('Context_25','CAFEGOURMAND') )
    
    print (" The probability that a given user takes a 'CAFEGOURMAND' when he is in the Context_35 is:",
    		   "%.3f" % prob_ctx('Context_35','CAFEGOURMAND') )
    
    
    	# Making a matrix storing all probabilities p(Product| a given context)
    l=list(itertools.product(PrCAmatrix['context'].values, PrCAmatrix.columns.values[1:]))
    
    for s in [2,3,4]:
        r=(prob_ctx(l[s][0],l[s][1]))
    print(r.shape)
    print(r)
    
    for s in np.arange(len(l)):
        m=np.array((prob_ctx(l[s][0],l[s][1])))
    print(m)
    
    #!!! another method to compute P(Product| a given context):
    
    ((x,y) for x in PrCAmatrix['context'].values for y in PrCAmatrix.columns.values[1:])
    M_prob=[prob_ctx(x,y) for x in  PrCAmatrix['context'].values for y in PrCAmatrix.columns.values[1:]]
    M_prob=np.array(M_prob).reshape(50,20)
    M_prob=pd.DataFrame(M_prob, columns= PrCAmatrix.columns.values[1:],index=PrCAmatrix['context'].values)
    
    #==============================================================================
    #SIMILARITY
    # Now, we will apply heuristic approach to try to make sense to relation between User and Item.
    
    	# For that, we need 2 matrix: the first will be used for describing the Item's profile and the 2nd for the User's profile.
    	# 
    	# The similarity between an given user and an given Item will be computed by Cosinus formula ( or another technique, we will try to do that with different similarity based computation methods
    	# 
    
    	# ITEM'S PROFILE
    	# 
    	# For this simulating work, we will take 6 attributs to describe an Item's profile, so a given item will be presented by:
    	# - age_buyer = the weighted average of ages of users who have bought this item ( age*number of transactions made by this age/ sum of transactions)
    	# - selling_price: of this item
    	# - man_buyer: number of men who bought this item/ total of buyers
    	# - woman_buyer: number of women who bought this item/ total of buyers
    	# - terrace_buyer: number of times this item have been bought when users were in terrace/ total of solded items
    	# - inside_buyer: number of times this item have been bought when users were not in terrace=inside (??)/ total of solded items
    
    #==============================================================================
    
    # Generating a list containing all users' age for each item ( 20 products)** note: better with list and not DF
    np.random.seed(4)
    n=np.random.randint(6,90,30)
    n1=np.random.randint(20,75,30)
    n2=np.random.randint(5,17,30)
    age_buyer={'CAFEGOURMAND': np.random.choice(n,20,replace=True),
    			   'MUFFINS':np.random.choice(n,30,replace=True),
    			  'VODKA':np.random.choice(n1,25,replace=True),
    			  'SCHWEPPES AGRUMES': np.random.choice(n,15,replace=True),
    			  'SALADE VERTE': np.random.choice(n,20,replace=True),
    			 'THE SD': np.random.choice(n,26,replace=True),
    			 'AVE CESAR TERRE': np.random.choice(n,31,replace=True),
    			'PLANTAMAX': np.random.choice(n,19,replace=True),
    			 'COMPOTEE CERISE NOIRE': np.random.choice(n2,40,replace=True),
    			   'THE SD1': np.random.choice(n,21,replace=True),# ATTENTION: adding 1 to differ from other TH SD
    			'SOUPEFRUIT': np.random.choice(n,33,replace=True),
    			'VERRE PAPOLLE BLANC SEC': np.random.choice(n,14,replace=True),
    			 'PELFORTH':np.random.choice(n1,55,replace=True),
    			  'MAX CUVEE': np.random.choice(n,12,replace=True),
    			   'PICHET PUNCH':np.random.choice(n1,47,replace=True),
    			   'DECA': np.random.choice(n,55,replace=True),
    			   'PAGO ORANGE': np.random.choice(n,31,replace=True),
    			  'MENU PLAT': np.random.choice(n,46,replace=True),
    			  'AVE CESAR TERRE SD': np.random.choice(n,37,replace=True),
    			  'XAMAX': np.random.choice(n,51,replace=True)
    			  }
    
    
    # compute user's weighted average age for each item
    avg={k:np.sum(v)/len(v) for k,v in age_buyer.items()}
    avg= pd.Series(avg)
    avg=pd.DataFrame(avg, columns=['avg_age'])
    #ax = avg.avg_age.plot(kind='bar')
    #ax.axis(ymin=0, ymax=70);
    
    # Using Standart Deviation & Mahanobis distance based method
    # Converting to a data frame  Users' age for each product
    age_buyer=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in age_buyer.items() ]))
    	
    # this plot can be served to match the distribution of buyers' age for each product
    #age_buyer.hist(figsize=[20,15]);
    
    #==============================================================================
    # We will use the Standard Deviation indicator to mesure the dispersion of values and how far they are from the population distribution (Mahanobis distance).
    # The standard deviation measure is based on the statistical concept of normal distribution, or a common expected shape of distribution among various types of data. The value for standard deviation defines a range above and below the mean for which a certain percentage of the data lie. 
    # So, we will need to make a standardization (or Z-score normalization) in which the features are rescaled so that they’ll have the properties of a standard normal distribution with μ=0 and σ=1, where μ is the mean (average) and σ is the standard deviation from the mean.
    # Firstly, we're going to build some math&statistical functions.
    # Some math & statistical functions
    #==============================================================================
    def dot(v, w):
        """v_1 * w_1 + ... + v_n * w_n"""
        return sum(v_i * w_i for v_i, w_i in zip(v, w))
    
    def sum_of_squares(v):
        """v_1 * v_1 + ... + v_n * v_n"""
        return dot(v, v)
    
    def median(v):
        """finds the 'middle-most' value of v"""
        v = [value for value in v if not math.isnan(value)]# Using this code to ignore null values
        n = len(v)
        sorted_v = sorted(v)
        midpoint = n // 2
        if n % 2 == 1:
            # if odd, return the middle value
            return sorted_v[midpoint]
        else:
            # if even, return the average of the middle values
            lo = midpoint - 1
            hi = midpoint
            return (sorted_v[lo] + sorted_v[hi]) / 2

    def mean(x):
        x = [value for value in x if not math.isnan(value)]# Using this code to ignore null values
        return sum(x) / len(x)
    
    def de_mean(x):
        """translate x by subtracting its mean (so the result has mean 0)"""
        x = [value for value in x if not math.isnan(value)]# Using this code to ignore null values
        x_bar= mean(x)
        return [x_i - x_bar for x_i in x]
    def variance(x):
        """assumes x has at least two elements"""
        x = [value for value in x if not math.isnan(value)]# Using this code to ignore null values
        n = len(x)
        deviations = de_mean(x)
        return sum_of_squares(deviations) / (n - 1)

    #variance(num_friends)
    def standard_deviation(x):
        x = [value for value in x if not math.isnan(value)]# Using this code to ignore null values
        return math.sqrt(variance(x))
    
    std_age=age_buyer.apply(standard_deviation,axis=0)
    mean_age=age_buyer.apply(mean,axis=0)
    median_age=age_buyer.apply(median,axis=0)
    age_stat=pd.DataFrame([mean_age,median_age,std_age]).T
    age_stat.columns=['mean_age','median_age','std_age']
    
    
    # Let's keep generating variables describing an Item's profile
    selling_price={'CAFEGOURMAND': 7.9,
    			   'MUFFINS':6.9,
    			  'VODKA':4.5,
    			  'SCHWEPPES AGRUMES': 3.9,
    			  'SALADE VERTE': 5.9,
    			 'THE SD': 2.9,
    			'THE SD1': 3.9,
    			 'AVE CESAR TERRE': 10.9,
    			'PLANTAMAX': 12.9,
    			 'COMPOTEE CERISE NOIRE': 7.9,
    			 'SOUPEFRUIT': 8.9,
    			'VERRE PAPOLLE BLANC SEC': 5.9,
    			 'PELFORTH':5.1,
    			  'MAX CUVEE': 15.9,
    			   'PICHET PUNCH':9.9,
    			   'DECA': 3.5,
    			   'PAGO ORANGE': 3.9,
    			  'MENU PLAT': 19.9,
    			  'AVE CESAR TERRE SD': 12.9,
    			  'XAMAX': 13.9
    			  }
    man_buyer={'CAFEGOURMAND': np.random.choice(n),
    			   'MUFFINS':np.random.choice(n),
    			  'VODKA':np.random.choice(n),
    			  'SCHWEPPES AGRUMES': np.random.choice(n),
    			  'SALADE VERTE': np.random.choice(n),
    			 'THE SD': np.random.choice(n),
    			   'THE SD1': np.random.choice(n),
    			 'AVE CESAR TERRE': np.random.choice(n),
    			'PLANTAMAX': np.random.choice(n),
    			'COMPOTEE CERISE NOIRE':np.random.choice(n),
    			   'SOUPEFRUIT': np.random.choice(n),
    			'VERRE PAPOLLE BLANC SEC': np.random.choice(n),
    			 'PELFORTH':np.random.choice(n),
    			  'MAX CUVEE': np.random.choice(n),
    			   'PICHET PUNCH':np.random.choice(n),
    			   'DECA': np.random.choice(n),
    			   'PAGO ORANGE': np.random.choice(n),
    			  'MENU PLAT': np.random.choice(n),
    			  'AVE CESAR TERRE SD': np.random.choice(n),
    			  'XAMAX': np.random.choice(n)}
    woman_buyer={'CAFEGOURMAND': np.random.choice(n),
    			   'MUFFINS':np.random.choice(n),
    			  'VODKA':np.random.choice(n),
    			  'SCHWEPPES AGRUMES': np.random.choice(n),
    			  'SALADE VERTE': np.random.choice(n),
    			 'THE SD': np.random.choice(n),
    				 'THE SD1': np.random.choice(n),
    			 'AVE CESAR TERRE': np.random.choice(n),
    			'PLANTAMAX': np.random.choice(n),
    			'COMPOTEE CERISE NOIRE':np.random.choice(n),
    				 'SOUPEFRUIT': np.random.choice(n),
    			'VERRE PAPOLLE BLANC SEC': np.random.choice(n),
    			 'PELFORTH':np.random.choice(n),
    			  'MAX CUVEE': np.random.choice(n),
    			   'PICHET PUNCH':np.random.choice(n),
    			   'DECA': np.random.choice(n),
    			   'PAGO ORANGE': np.random.choice(n),
    			  'MENU PLAT': np.random.choice(n),
    			  'AVE CESAR TERRE SD': np.random.choice(n),
    			  'XAMAX': np.random.choice(n)}
    terrace_buyer={'CAFEGOURMAND': np.random.choice(n),
    			   'MUFFINS':np.random.choice(n),
    			  'VODKA':np.random.choice(n),
    			  'SCHWEPPES AGRUMES': np.random.choice(n),
    			  'SALADE VERTE': np.random.choice(n),
    			 'THE SD': np.random.choice(n),
    				   'THE SD1': np.random.choice(n),
    			 'AVE CESAR TERRE': np.random.choice(n),
    			'PLANTAMAX': np.random.choice(n),
    			'COMPOTEE CERISE NOIRE':np.random.choice(n),
    			 'SOUPEFRUIT': np.random.choice(n),
    			'VERRE PAPOLLE BLANC SEC': np.random.choice(n),
    			 'PELFORTH':np.random.choice(n),
    			  'MAX CUVEE': np.random.choice(n),
    			   'PICHET PUNCH':np.random.choice(n),
    			   'DECA': np.random.choice(n),
    			   'PAGO ORANGE': np.random.choice(n),
    			  'MENU PLAT': np.random.choice(n),
    			  'AVE CESAR TERRE SD': np.random.choice(n),
    			  'XAMAX': np.random.choice(n)}
    inside_buyer= {'CAFEGOURMAND': np.random.choice(n),
    			   'MUFFINS':np.random.choice(n),
    			  'VODKA':np.random.choice(n),
    			  'SCHWEPPES AGRUMES': np.random.choice(n),
    			  'SALADE VERTE': np.random.choice(n),
    			 'THE SD': np.random.choice(n),
    				   'THE SD1': np.random.choice(n),
    			 'AVE CESAR TERRE': np.random.choice(n),
    			'PLANTAMAX': np.random.choice(n),
    			'COMPOTEE CERISE NOIRE':np.random.choice(n),
    				'SOUPEFRUIT': np.random.choice(n),
    			'VERRE PAPOLLE BLANC SEC': np.random.choice(n),
    			 'PELFORTH':np.random.choice(n),
    			  'MAX CUVEE': np.random.choice(n),
    			   'PICHET PUNCH':np.random.choice(n),
    			   'DECA': np.random.choice(n),
    			   'PAGO ORANGE': np.random.choice(n),
    			  'MENU PLAT': np.random.choice(n),
    			  'AVE CESAR TERRE SD': np.random.choice(n),
    			  'XAMAX': np.random.choice(n)}
    It_PROFILE=pd.DataFrame({'selling_price':selling_price,'man_buyer':man_buyer,'woman_buyer':woman_buyer,'terrace_buyer':terrace_buyer,
    					   'inside_buyer':inside_buyer})
    
    #Merging "age_stat" and "It_PROFILE"
    IT_PF=pd.merge(age_stat,It_PROFILE, right_index=True,left_index=True)
    	
    # We need to convert these features in the way they help us to mesure the similarity between an given user and an given item. This feature engineering task also aims to make more relevant features we have.
    
    IT_PF['inside_buyertf']= IT_PF['inside_buyer']/(IT_PF['inside_buyer']+IT_PF['terrace_buyer'])
    IT_PF['terrace_buyertf']= IT_PF['terrace_buyer']/(IT_PF['inside_buyer']+IT_PF['terrace_buyer'])
    IT_PF['man_buyertf']= IT_PF['man_buyer']/(IT_PF['man_buyer']+IT_PF['woman_buyer'])
    IT_PF['woman_buyertf']= IT_PF['woman_buyer']/(IT_PF['man_buyer']+IT_PF['woman_buyer'])
    
    
    # For computing the similarity between an Item and an User, we also need a vector describing the User's profile
    # In this simulation, we will use the data set created in the scrip "SimulBase_Aproache1_Classification'
    
    # Let's import this data set
    
    USER=pd.read_csv('C:/Users/Pham Antoine/source/repos/ConsoleApp1/PythonApplication1/USER_simul.csv',sep=',')
    USER=USER.iloc[:,1:]# to drop the 1st column ( index), to avoid that , when export to csv put " index=False"
    	
    # We have to do some manipulations so that our cosinus mesure can work
    # Transforming some orignal variables to dummies variables ( pd.get_dummies will not work because these variables are numerical)
    
    USER['inside_buyertf']=1
    USER.loc[USER.user_terrace==1,'inside_buyertf']=0
    USER['terrace_buyertf']= 1 
    USER.loc[USER.user_terrace==0,'terrace_buyertf']=0
    
    # ...and more conversion to dummies variables
    
    USER['man_buyertf']=1
    USER['man_buyertf']=np.where(USER['user_genre']==0,0,USER['man_buyertf'])
    
    def f(row):
        if row['user_genre']==0:
            val= 1
        else:
            val=0
            return val
    USER['woman_buyertf']=USER.apply(f,axis=1)
    
    User_cos=USER.iloc[:,8:]
    #==============================================================================
    # How to scale continous variables?
    # Scaling the variable ' avg_ticketU'- remember that this action must be applied to the variable 'selling_price' in the Item's profile dataset
    # This action is important but may be more important in the situation where we have multiple continous variables (in our case , there's only one variable of this kind)
    # There are several techniques to scale a variable: normalization by (valeur-min/max-min); substracting the mean then dividing by the standart deviation (mahanobis) or by adding an alpha factor as disscused by Lescovec etal. (p.316)... and we should choose the best one...
    #=============================================================================
    # Here, we try to scale by adding an alpha = 0.1( Later, we will aplly Mahanobis method to detect the similarity between 'avg_ticketU' and 'selling price').
    
    alpha=0.1
    USER['ticket_scaled']=USER['avg_ticketU']*alpha# alpha= 0.1 ( see Lescovec p.316)
    # Doing the same action for Item's price
    
    # Creating a new variable that is scaled by the variable 'selling_price' , we give the name 'ticket_scaled to this new variable just for having the same name with the component in the User's vector 
    IT_PF['ticket_scaled']= IT_PF['selling_price']*alpha
    
    IT_cos=IT_PF.iloc[:,8:]
    #==============================================================================
    # The Similarity.
    # We're goining to use 2 methods to compute for each pair User/ Item a value (SIM) allowing us to obtain the similartity between the User and the Item in question . By the way, we will be also able to make a ranking either User/ nearest Items or Item/nearest Users.
    # - SIM= Cos_dist + Mah_dist
    # where :
    # - Cos_dist = Cosine distance between 2 vectors folowing:
    # 1) Users ( inside_buyertf, terrace_buyertf, man_buyertf, ticket_scaled) 
    # 2) Items (inside_buyertf, terrace_buyertf, man_buyertf, woman_buyertf, selling_price)
    # - Mah_dist= Mahalanobis distance between 'user_age' to the normalized distribution refleting the components 'age' of a product. This feature will be calculed by the Z_score metric( the lower this metric is  , the more the user is "similar" to the item)
    #==============================================================================
     # Cosine distance
    def square_rooted(x):
        return round(math.sqrt(sum([a*a for a in x])),3)
    def cosine_similarity(x,y):
        numerateur = sum(a*b for a,b in zip(x,y))
        denominateur = square_rooted(x)*square_rooted(y)
        return round(numerateur/float(denominateur),3)

   # Adding a term "User_N°..." as index . This action will be useful for mesuring cosine distance later
    idx = ['User_%d' % i for i in range(1,21)]
    User_cos.index=idx
    
    # Calculate the cosine mesure
    ((x,y) for x in User_cos.values for y in IT_cos.values)
    Cos=[cosine_similarity(x,y) for x in User_cos.values for y in IT_cos.values]
    Cos=np.array(Cos).reshape(20,20)# shape of an array 20x19 ( number of users * number of products)
    Cos=pd.DataFrame(Cos,columns= IT_cos.index.values,index = User_cos.index.values)
    
    print("Checking by an example : What is the cosine distance between the User 19 and 'Salade verte'?")
    print("By using the Cosine based distance function, this distance is:")
    print(round(cosine_similarity(User_cos.loc['User_19'] ,IT_cos.loc['SALADE VERTE']),3))
    print(" ... and by the matrix 'Cos' already cretead:")
    print(Cos.loc['User_19','SALADE VERTE'])
    
    # Mahalanobis distance
    
    # Let's build a function computing the Mahalanobis distance between a given point (user's age) to a given distribution ( ave_age of an given Item). Attention : this is not a function computing the classical Mahalanobis which mesures a distance between 2 distributions.
    
    def mahalanobis_similarity(x,y,z):
        numerateur = (x-y)# x= age_user; y= avg_age of Item
        denominateur = z # z = standart deviation of age Item
        return round(math.sqrt(pow(numerateur/denominateur,2)),3)
    print("The Mahalanobis distance between the User 1 and 'AVE CESAR TERRE' is: ")
    print(mahalanobis_similarity(63,51.161290,20.210718))
    print(" ... and the Mahalanobis distance from the User 2 to 'AVE CESAR TERRE' is: ")
    print(mahalanobis_similarity(42,51.161290,20.210718))
    
    l1= list(np.array(IT_PF.loc[:, ['mean_age','std_age']]))# to get all pairs of y,z
    for y,z in l1:
        print(y,z)           
    
    #((x,y,z) for x in USER['user_age'].values for y,z in l1)
    idx = ['User_%d' % i for i in range(1,21)]
    Mah=[mahalanobis_similarity(x,y,z) for x in USER['user_age'].values for y,z in l1]
    Mah=np.array(Mah).reshape(20,20)# shape of an array 20x19 ( number of users * number of products)
    Mah=pd.DataFrame(Mah,columns= IT_PF.index.values,index = idx)
    
    #==============================================================================
    #  Let's make predictions
    # Given an User with profile (i)  in the context (c) for each item, making a
    # Score = W1** prob_ctx(context,product) + W2*** cosine_similarity(x,y) + (1-W3 * mahalanobis_similarity(x,y,z))
    # The higher this score is, the more the probability the User buy the product will be important. That's why we've added: 1-(1-W3 * mahalanobis_similarity(x,y,z)) instead of W3 * mahalanobis_similarity(x,y,z) to get the same direction of the score.
    # We first initialize the parameters  w1,w2,w3 as folowing:
    #==============================================================================
    # Initializing of parameters (w)
    w1,w2,w3=[0.5,0.25,0.5] # these hyper-parameters can be tuned
    
    #==============================================================================
    # We suppose a new User with the  components of his profile's vector as following: 
    # It's a man seeting inside, we also know he spends around 13 euros per visite
    #==============================================================================
    # inside_buyertf=1;terrace_buyertf=0;man_buyertf=1;woman_buyertf=0;ticket_scaled=13
    user_lamda=[1,0,1,0,0]
    
    # The customer has 15 years old
    user_lamda_age=25
    # We are on the context 24 at the moment  
    Context='Context_24'
    # For instance, if we'd love to get the score about the relation between this User with the Item 'Cafe Gourmand' 
    print("Score for each factor of the model:")
    print(round(w1*prob_ctx(Context,'CAFEGOURMAND'),4))
    print(w2*cosine_similarity(user_lamda,IT_cos.loc['CAFEGOURMAND'])) # to mesure imilarity between 2 vectors: Users and Item profile
    print(round(1-w3*mahalanobis_similarity(user_lamda_age,IT_PF.loc['CAFEGOURMAND','mean_age'],
    									  IT_PF.loc['CAFEGOURMAND','std_age']),4))
    #mahalanobis_similarity(x,y,z); # y, z = matching item's values  in the IT_PF matrix=> 'CAFEGOURMAND': y=avg_age, z=std_age
    print("Total score given by this new User on Context 46 related to the Item ' Cafe Gourmand :")
    print(round(w1*prob_ctx(Context,'CAFEGOURMAND')+w2*cosine_similarity(user_lamda,IT_cos.loc['CAFEGOURMAND'])
    		 +(1-w3*mahalanobis_similarity(user_lamda_age,IT_PF.loc['CAFEGOURMAND','mean_age'],IT_PF.loc['CAFEGOURMAND','std_age'])),5))
    #print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores_grouped1.mean(), scores_grouped1.std(), label))
    
    # Making Rating Scores
    C_play=[prob_ctx(Context,y) for y in PrCAmatrix.columns.values[1:]]
    C_play=pd.Series(C_play)*w1
    
    Simcos_play=[cosine_similarity(user_lamda, y) for y in IT_cos.values]
    Simcos_play=pd.Series(Simcos_play)*w2
    
    Simmah_play=[mahalanobis_similarity(user_lamda_age,y,z)  for y,z in l1]
    Simmah_play=1-pd.Series(Simmah_play)*w3
    
    # Filtering products not allowed to recommend to minor
    
    alcollist=['KIR','KIR ROYAL','PUNCH Maison','SANGRIA','RHUM ARRANGE','PICHET PUNCH',
    				   'PICHET SANGRIA','WHISKY','RHUM','VODKA','GIN','GET','TEQUILA','MALIBU','BAILEYS','RICARD',
    				   'PASTIS','PINEAU','PORTO','LILLET','MARTINI','CHAMPAGNE BOUTEILLE','CHAMPAGNE COUPE','PELFORTH',
    				   'PINT PELFORTH','GRIMBERGEN','HOEGGARDEN','DESPERADOS','KEKETTE EXTRA','KEKETTE RED',
    				   'BIERE SANS GLUTEN','EXPRESSO','DECA','DOUBLE EXPRESSO','PAPOLLE BLANC SEC','VERRE PAPOLLE BLANC SEC',
    		   'PAPOLLE BLANC MOEL','VERRE PAPOLLE BLANC MOEL','PAPOLLE ROSE','VERRE PAPOLLE ROSE','BIERE PRESSION','EXPRESSO SD',
    		   'DECA SD','THE SD','INFUSION SD','BIERE',' PRESSION MAX MENU','BIERE PINTH','VODKA','PELFORTH','VERRE PAPOLLE ROSE','THE SD1','THE SD','PICHET PUNCH','VERRE PAPOLLE BLANC SEC']
    
    Ranking=(C_play+Simcos_play+Simmah_play)
    Ranking.index=IT_PF.index
    Ranking=pd.DataFrame(Ranking.sort_values(ascending=False),columns=['Score'])
    top=10
    def k(df):
        if user_lamda_age >18:
            return df.head(top)
        else:
            d=Ranking[~Ranking.index.isin(alcollist)]# filtering products being in the list we have created above
            return d.head(top)
    Recommendation_list=k(Ranking)
    print(" Voici la list de produits à proposer au client:")
    print(Recommendation_list)
    
    f, ax = plt.subplots(figsize=(16,10))
    fig = sns.boxplot(y=Recommendation_list.index, x=Recommendation_list['Score']) # making a Box Plot to observe average
    print(fig);
        
    #another way to do that by converting the result to a dictionary
    Recommendation_list.T.to_dict()
      
    sorted(Recommendation_list.items(), key=operator.itemgetter(0),reverse=False)
    #print (__name__)
    print (" TEST HERE!!")
    
    return Recommendation_list
 #newEvent.Parameter, newEvent.TimeStamp,newEvent.NbPersons,newEvent.UserID

if __name__== "__main__":
    JOB()
else:
    print("Run from import")
    	
def Ft(AGEL):
	return AGEL+150