from hdp_driver import preprocess, Process, TFIDF, HDP

seed_filename = 'pgp_seed.txt'
stream_filename = 'pgp_rest.txt'

#Remove the top 5% TF-IDF words from the corpus.
#If no trim_top is specified, it will be set to 2,5% by default.
seed = preprocess(seed_filename,trim_top=5)
stream = preprocess(stream_filename,trim_top=5)

#Run HDP and discard topics with alpha below 0.007.
hdp = HDP(seed,stream,threshold=0.007)
hdp.build_hdp()

#Print how many topics we've got.
hdp.update_topics()
print(len(hdp.topics))

#Create a spreadsheet with the top 40 words from each topic, one topic per line.
hdp.print_table(topn=40,filename='data.csv')

#Identify the topics in a document with HDP.
tpc1 = hdp.build_topics('test1.txt')
tpc2 = hdp.build_topics('test2.txt')

#Plot how much the document fits into each topic.
hdp.plot(tpc1,'test1.png')
hdp.plot(tpc2,'test2.png')


