# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:32:06 2017

@author: brucesmith
"""
#THIS IS PROTYPE CODE FOR SVD AND KMEANS THAT PRODUCES tHE SPRINT 3 RESULTS
#IN THE CODE ORDEr HAS BEEN CHANGED TO PROTECT THE INNOCENT
#READ IN DATA FROM ARTICLE DB AND STORE HEADER AND TEXT
import sqlite3
con=sqlite3.connect('articles.db') #connect to sqlite db
curse=con.cursor() #invoke cursor as curse
#curse.execute('select * from articles') #select all the articles IF DO QUERIES USE ? FORM
curse.execute('select * from articles LIMIT 5 OFFSET 11') #select all the fields from 11 to 15
#get column names from description of cursor, description is 7-tuple with last 6 are None
names = tuple(description[0] for description in curse.description) #column names
print names
#row=curse.fetchone() #get one row CONSIDER SQLITE ROW FACTORY HERE
#print row
row_count=5
artext=[] #article text list to be analyzed NEED TO create A PLACE FOR OTHER ROW COLUMNS
urls=[] #urls
#curse.execute('select title,url,publish_date from articles LIMIT 5 OFFSET 11')
curse.execute('select * from articles LIMIT 5 OFFSET 11') #select all the articles IF DO QUERIES USE ? FORM
for r in range(0,row_count):
    row=curse.fetchone()
    print r+11,row[0],row[3],row[2]
    urls.append(row[3])
    artext.append(row[4]) #row[4] is the text field in utf-8 format
con.close()


#read in url label assignments
import csv
urllabels={}
with open('labels.csv','rU') as labelfile:
    myReader=csv.reader(labelfile)
    for row in myReader:
        url=row[0]
        label=row[1]
        urllabels[url]=label

urlkeys=[]
#fetch labels for article urls      
for r in range(0,row_count):
    url=urls[r]
    lurl=url.lstrip('htps:/')
    lurlr=lurl.rsplit('/')
    urlkey=str(lurlr[0])
    label=urllabels[urlkey]
    print urlkey,label
    urlkeys.append([urlkey,label])
    
import math
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvect = TfidfVectorizer() #like countvectorizer but tfidf instead
tfidf_matrix=tfidfvect.fit_transform(artext)
tf_features=tfidfvect.get_feature_names()
words=tf_features

dense = tfidf_matrix.todense() #SAME AS toarray?
import numpy as np
import scipy
from matplotlib import pyplot as plt

A=dense.transpose()

ATA=dense*A
svd=scipy.linalg.svd(A)


S=svd[0]
Sigma=svd[1]
UT=svd[2]

docvecs=Sigma*UT

#obtain word and document coordinates in eigenspace
wordcoords=[]
windex=0
ndim=3
numwords=len(words)
for w in words:
    coords=[]
    for k in range(0,ndim):
        coord=S[windex][k]*Sigma[k]
        coords.append(coord)
    wordcoords.append([w,coords])
    windex=windex+1
windex=0
wordvecs=[]
for w in words:
    coords=[]
    magsq=0
    for k in range(0,5):
        coord=S[windex][k]*Sigma[k]
        coords.append(coord)
        csq=coord*coord
        magsq=magsq+csq
    wordvecs.append([w,coords,magsq])
    windex=windex+1

normtotal=0
normsqtotal=0
for w in range(0,numwords):
        norm=math.sqrt(wordvecs[w][2])
        normtotal=normtotal+norm
        normsqtotal=normsqtotal+wordvecs[w][2]
normavg=normtotal/numwords
normavg2=normavg*normavg
magsqnorm=normsqtotal/numwords

numdocs=len(UT)
dcoords=[]
doccoords=[]
for d in range(0,numdocs):
    coords=[]
    for k in range(0,ndim):
        coord=Sigma[k]*UT[k][d]
        coords.append(coord)
    doccoords.append([d+1,coords])
    dcoords.append(coords)

docvecs=[]
for d in range(0,numdocs):
    coords=[]
    magsq=0
    for k in range(0,5):
        coord=Sigma[k]*UT[k][d]
        coords.append(coord)
        csq=coord*coord
        magsq=magsq+csq
    docvecs.append([d+1,coords,magsq])
    
csumtotal=0
csums=[]
for w in range(0,numwords):
    sum=0
    for d in range(0,numdocs):    
        sum=A[w,d]+sum
    csums.append(sum)
    csumtotal=csumtotal+sum

#combine word and document coords by axis
xcoords=[]
ycoords=[]
zcoords=[]
xindex=1
yindex=2
zindex=0

for w in range(0,numwords):
    norm=math.sqrt(magsqnorm)
    xcoords.append(wordcoords[w][1][xindex]/norm)
    ycoords.append(wordcoords[w][1][yindex]/norm)
    zcoords.append(wordcoords[w][1][zindex]/norm)
    
for d in range(0,numdocs):
    xcoords.append(doccoords[d][1][xindex])
    ycoords.append(doccoords[d][1][yindex])
    zcoords.append(doccoords[d][1][zindex])

#labels for x,y coordinates
tlabels=[]
for w in range(0,numwords):
    tlabels.append(wordcoords[w][0])
for d in range(0,numdocs):
    tlabels.append(doccoords[d][0])
plt.clf()    
ax=plt.axes()
ax.set_xlim([-2.5,2.5])
ax.set_ylim([-2.5,2.5])
plt.grid()
plt.scatter(xcoords,ycoords)

for t in range(numwords,len(xcoords)):
    plt.annotate(tlabels[t],xy=(xcoords[t],ycoords[t]))
    
featurecoords=[]

for t in range(numwords,len(xcoords)):
    featurecoords.append([xcoords[t],ycoords[t]])
features=np.array(featurecoords)

from scipy.cluster.vq import kmeans,vq
from scipy.spatial import Voronoi, voronoi_plot_2d

numclusters=4
km=scipy.cluster.vq.kmeans(dcoords,numclusters,iter=25)
km
code_book=km[0]
assignments=vq(dcoords,km[0])
for t in range(1292,len(xcoords)):
    print tlabels[t],assignments[0][t-numwords]

xcentroids=[]
ycentroids=[]
zcentroids=[]
for n in range(0,numclusters):
    xcentroids.append(code_book[n][xindex])
    ycentroids.append(code_book[n][yindex])
    zcentroids.append(code_book[n][zindex])
    #ax.arrow(0,0,xcentroids[n],ycentroids[n])
clabels=np.arange(0,numclusters)
for c in range(0,numclusters):
    plt.annotate(clabels[c],xy=(xcentroids[c],ycentroids[c]),color='r')


plt.scatter(xcentroids,ycentroids,color='r',marker='o')
    
ax=plt.gca()
vorpoints=[]
for c in range(0,numclusters):
    point=[code_book[c][xindex],code_book[c][yindex]]
    vorpoints.append(point)
vor = Voronoi(vorpoints)
voronoi_plot_2d(vor,ax)
"""
for region in vor.regions:
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon))
"""
plt.plot(featurecoords)
plt.show()


numfeatures=len(xcoords)
cdots=[]
for c in range(0,numclusters):
    cdot=[]
    for f in range(0,numfeatures):
        dx=xcentroids[c]-xcoords[f]
        dy=ycentroids[c]-ycoords[f]
        dz=zcentroids[c]-zcoords[f]
        centdotfeat=math.sqrt(dx*dx+dy*dy+dz*dz)
        cdot.append(centdotfeat)
    cdots.append(cdot)


fmagsqs=[]
for f in range(0,numfeatures):
    dx=xcoords[f]
    dy=ycoords[f]
    dz=zcoords[f]
    fmagsq=dx*dx+dy*dy+dz*dz
    fmagsqs.append(fmagsq)

wordmagsqtot=0
for w in range(0,numwords):
    wordmagsqtot=wordmagsqtot+fmagsqs[w]
    
docmagsqtot=0
for d in range(numwords,numfeatures):
    docmagsqtot=docmagsqtot+fmagsqs[d]


topfew=10
for c in range(0,numclusters):
    centfeats=np.argsort(cdots[c])
    for f in range(0,10):
        order=centfeats[f]
        plt.annotate(tlabels[order],xy=(xcoords[order],ycoords[order]),color='g')
        print c,order, tlabels[order],xcoords[order],ycoords[order], cdots[c][order]
"""
for f in range(0,10):
        order=centfeats[f]
        plt.annotate(tlabels[order],xy=(xcoords[order],ycoords[order]),color='g')
        print c,order, tlabels[order],xcoords[order],ycoords[order], cdots[c][order]
"""
dmag=[]
for d in range(0,numdocs):
    mag=0    
    for n in range(0,ndim):
        mag=mag+dcoords[d][n]*dcoords[d][n]
    dmag.append(mag)

for c in range(0,numclusters):
    centfeats=np.argsort(cdots[c])
    for f in range(0,topfew):
        order=centfeats[f]
        print c,tlabels[order],cdots[0][order],cdots[1][order],cdots[2][order],cdots[3][order]
 

for c in range(0,numclusters):
    centfeats=np.argsort(cdots[c])
    for f in range(0,topfew):
        mag=0
        order=centfeats[f]
        for n in range(0,ndim):
            if order<numwords:
                mag=wordcoords[order][1][n]*wordcoords[order][1][n]+mag
            else:
                d=order-numwords
                mag=mag+dcoords[d][n]*dcoords[d][n]
        print c,tlabels[order],cdots[0][order],cdots[1][order],cdots[2][order],cdots[3][order]
        #print c,order,tlabels[order],mag,cdots[c][order]

xymags=[]
for w in range(0,numwords):
    xymag=xcoords[w]*xcoords[w]+ycoords[w]*ycoords[w]
    xymags.append(-xymag)    
xyindex=np.argsort(xymags)
for t in range(0,50):
    order=xyindex[t]
    mag=math.sqrt(-xymags[order])
    print order,words[order],xcoords[order],ycoords[order],mag
    plt.annotate(tlabels[order],xy=(xcoords[order],ycoords[order]),color='r')

plt.title('K-means Clusters')
plt.xlabel('SV2')
plt.ylabel('SV3')
import matplotlib.patches as mpatches
green_patch = mpatches.Patch(color='green', label='cluster words')
red_patch = mpatches.Patch(color='red', label='outlier words')
plt.legend(handles=[green_patch,red_patch])

plt.show()
    
import csv
coordWriter=csv.writer(open('coords.csv','wb'))
for d in range (0,numdocs):
    coords=[d+1,doccoords[d][1][0],doccoords[d][1][1]]
    coordWriter.writerow(coords)

index=0
for w in words:
    coords=[w,wordcoords[index][1][0],wordcoords[index][1][1]]
    coordWriter.writerow(coords)
    index=index+1

for d in range(0,numdocs):
    doccoords[d][1]
    print urlkeys[d][0],assignments[0][d]