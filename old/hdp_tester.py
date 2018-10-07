from hdp_driver import build_data, DF, HDP

seed = DF('pgp_seed.csv',['title','text'],threshold=0.01)
seed.run()
stream = DF('pgp_rest.csv',['title','text'],threshold=0.01)
stream.run()

dct, corpus = build_data(seed.df['tokenized'],stream.df['tokenized'])

hdp = HDP(corpus,dct,stream.df)
hdp.build_lda()
hdp.build_topic_dist()

similar = hdp.similarity_query(300)

print('QUERY: \n')

print(stream.df['text'][300])

print('SIMILAR LETTERS: \n')

for index in similar:
	print('=== \n')
	print(stream.df['text'][index])
