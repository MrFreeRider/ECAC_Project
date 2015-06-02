# Load required libraries -------------------------------------------------
libs <- c("NLP", "tm", "plyr", "class", "rpart", "e1071", "igraph", "RColorBrewer", "ggplot2", "slam", "topicmodels", "wordcloud", "SnowballC", "graph", "Rgraphviz", "fpc", "stringi", "lda", "stats", "proxy", "maptpx", "reshape2", "ggvis")
lapply(libs, require, character.only= TRUE)

################################################################################
# Set the path to the corpus directory.                                        #
# ---------------------------------------------------------------------------- #
setwd('~/Dropbox/mygitrepository/ECAC_Project')
# ---------------------------------------------------------------------------- #
# Load the corpus data.                                                        #
# ---------------------------------------------------------------------------- #
load.data <- read.csv("CSV_All_References.csv", header=TRUE, stringsAsFactors = FALSE, fileEncoding="UTF-8")
#source('topmod.r')
#source('corpus.r')
#source('lda.r')
#source('plotlda.r')
# Load the corpus data ----------------------------------------------------
#paper.corpus <- Corpus(DataframeSource(load.data))
#paper.corpus <- ds
paper.corpus <- Corpus(VectorSource(load.data$Abstract))
#paper.corpus$Names <- c(names(load.data))
paper.corpus <- na.omit(paper.corpus)

# Clean up the data -------------------------------------------------------
#Once we have a corpus, now the next step is modify the documents in it, e.g., stemming, stopword removal, et cetera. All this funcionality is named as transformation concept. In general, all transformations work are done to all elements of the corpus applying the tm_map() function. Next, we make some adjustments to the text; making everything lower case, removing punctuation, removing numbers, and removing common English stopwords. The 'tm_map' function allows us to apply transformation functions to a corpus.
toSpace <- content_transformer(function(x,pattern) gsub(pattern, " ", x))
corpus.m <- tm_map(paper.corpus, toSpace, "/|@|\\|")
corpus.m <- tm_map(corpus.m, content_transformer(tolower))
corpus.m <- tm_map(corpus.m, removePunctuation)
removeUnicode <- function(x) stri_replace_all_regex(x,"[^\x20-\x7E]","")
corpus.m <- tm_map(corpus.m, content_transformer(removeUnicode))
corpus.m <- tm_map(corpus.m, removeNumbers)
corpus.m <- tm_map(corpus.m, removeWords, stopwords("english"))
corpus.m <- tm_map(corpus.m, removeWords, c("title", "abstract", "method", "apparatus", "include", "disposed", "end", "thereof", "lower", "upper", "using", "used", "propose", "can", "also", "provid", "present", "system", "based", "device", "invention", "use", "approach", "problem", "develop", "implement", "paper", "techniqu"))
library(SnowballC)
corpus.m <- tm_map(corpus.m, stemDocument, language = "english")
corpus.m <- tm_map(corpus.m, stripWhitespace)

# Build the document term matrix ------------------------------------------
# Now we can actually begin to analyze the text. First, we create something called a Document Term Matrix (DTM) which create incidence matrix for words in a document.  For many methods of text analysis, specifically the so-called bag-of-word approaches, the common data structure for the corpus is a document-term matrix (dtm). This is a matrix in which the rows represent documents and the columns represent terms (e.g., words, lemmata). The values represent how often each word occured in each document. If your data is in common, unprocessed textual form (such as this document) then a good approach is to use the tm package. tm offers various ways to import texts to the tm corpus structure. Additionally, it offers various functions to pre-process the data. For instance, removing stop-words, making everything lowercase and reducing words to their stem.
corpus.dtm <- DocumentTermMatrix(corpus.m, control = list(minWordLength = 3))#, weighting = function(x)  weightTfIdf(x, normalize = FALSE)))
corpus.dtm.m <- as.matrix(corpus.dtm)
# Get Results -------------------------------------------------------------
# Below we only show the first 13 words and their frequencies in each document (i.e. for us, each 'document' is a paper selected from a repository). Next, we can begin to explore the DTM to find which words were used most. Below we specify that we want terms/words which were used 300 or more times (in all documents). findFreqTerms() function is applied to find the most occured terms on this matrix.
(freq.terms <- findFreqTerms(x = corpus.dtm, lowfreq = 300, highfreq = Inf))
maxterms<-apply(corpus.dtm, 1, which.max)
unique(corpus.dtm$dimnames$Terms[maxterms])

# Finding words which 'associate' together. Here, we are specifying the Term Document Matrix to use, the term we want to find associates for, and the lowest acceptable correlation limit with that term. This returns a vector of terms which are associated with 'comput' at 0.30 or more (correlation) -- and reports each association in decending order.
findAssocs(x = corpus.dtm, term = "parallel", corlimit = 0.2)

# If desired, terms which occur very infrequently (i.e. sparse terms) can be removed; leaving only the 'common' terms. Below, the 'sparse' argument refers to the MAXIMUM sparse-ness allowed for a term to be in the returned matrix; in other words, the larger the percentage, the more terms will be retained (smaller the percentage, the fewer (but more common) terms will be retained.
corpus.dtm.rs <- removeSparseTerms(x = corpus.dtm, sparse = 0.75)
sapply(corpus.dtm.rs, length)
dtmTotals <- apply(corpus.dtm.rs , 2, sum) #Find the sum of words in each Document
corpus.dtm.rs.m <- as.matrix(corpus.dtm.rs)
tm_term_score(corpus.dtm.rs, c("parallel", "speedup"))
MC_tokenizer(corpus.dtm.rs)
scan_tokenizer(corpus.dtm.rs)


#It can be useful to compute different metrics per term, such as term frequency, document frequency (how many documents does it occur), and td.idf (term frequency * inverse document frequency, which removes both rare and overly frequent terms). To make this easy, let's define a function term.statistics to compute this information from a document-term matrix (also available from the corpustools package)

#Compute a number of useful statistics for filtering words: term frequency, idf, etc. Not all terms are equally informative of the underlying semantic structures of texts, and some terms are rather useless for this purpose. For interpretation and computational purposes it is worthwhile to delete some of the less usefull words from the dtm before fitting the LDA model. We offer the term.statistics function to get some basic information on the vocabulary (i.e. the total set of terms) of the corpus.
freqs = term.statistics(corpus.dtm.rs)
freqs[sample(1:nrow(freqs), 10), ]

#We can now filter out words based on this information. In our example, we filter on terms that occur at least in two documents and that do not contain numbers. We also select only the terms with the highest tf-idf score (this is not a common standard. For large corpora it makes sense to include more terms). 
voca = as.character(freqs[order(freqs$tfidf, decreasing = T), ][1:length(freqs$term), "term"])
voca <- na.omit(voca)
filtered_dtm = corpus.dtm.rs[, voca] # select only the terms we want to keep

cmp = corpora.compare(corpus.dtm, corpus.dtm.rs)
cmp.train <- head(cmp, n=20)

cmp = cmp[order(cmp$over), ]
head(cmp)

#The list is very comparable, but more frequent terms are generally favoured in the chi-squared approach since the chance of 'accidental' overrepresentation is smaller. Let's make a word cloud of the most frequent negative terms:
neg = cmp[cmp$over < 1, ]
neg = neg[order(-neg$chi), ]
pal <- brewer.pal(6,"YlGnBu") # color model
wordcloud(neg$term[1:10], neg$chi[1:10], scale=c(6,.5), min.freq=1,
          max.words=Inf, random.order=FALSE, rot.per=.15, colors=pal)

pos = cmp[cmp$over > 1, ]
pos = pos[order(-pos$chi), ]
wordcloud(pos$term[1:10], pos$chi[1:10]^.5, 
          scale=c(6,.5), min.freq=1, max.words=Inf, random.order=FALSE, 
          rot.per=.15, colors=pal)

# Quantitative Analysis ---------------------------------------------------
# We can obtain the term frequencies as a vector by converting the document term matrix into a matrix and summing the column counts. By listing the most frequent terms ordered by frequencie. Generate Dictionary in wf
type.token.ratio <- length(unique(corpus.dtm.rs.m))/length(corpus.dtm.rs.m)
freqs.top <- sort(col_sums(filtered_dtm) , decreasing = TRUE)
pdf("./plots/1_MostFrequentWords_Pie.pdf")
pie(freqs$termfreq, labels = freqs$term)
dev.off()

pdf("./plots/1_MostFrequentWords_hbar.pdf")
ggplot(freqs, aes(freqs$term,freqs$termfreq)) + geom_bar(stat="identity") + xlab("Terms") + ylab("Count") + coord_flip()
dev.off()

pdf("./plots/1_MostFrequentWords_vbar.pdf")
colour <- heat.colors(length(freqs$term))
barplot (freqs.top, las=1, col = colour)
dev.off()

plot(filtered_dtm, terms=findFreqTerms(filtered_dtm, lowfreq=300)[1:10], corThreshold=0.02)
dev.off()

pdf("./plots/1_MostFrequentWords_cloud.pdf")
wordcloud(freqs$term[1:length(freqs$term)], freqs$termfreq[1:length(freqs$termfreq)], scale=c(8,.2), min.freq=1, max.words=Inf, random.order=FALSE, rot.per=.15, colors=brewer.pal(8,"Dark2"))
dev.off()

pdf("./plots/1_MostFrequentWords_dendrogram.pdf")
dist_matrix <- dist(freqs.top , method="manhattan")
fit <- hclust(dist_matrix)
plot(fit, lwd=1, lty=1, sub='', hang=-1, main = "Cluster Dendrogram of Most Frequent Terms", xlab="Frequented Terms", ylab = "DNM Distance")
rect.hclust (fit, 5)
dev.off()

words <- corpus.dtm.rs %>%
  as.matrix %>%
  colnames %>%
  (function(x) x[nchar(x) < 20])
length(words)
summary(nchar(words))

data.frame(nletters=nchar(words)) %>%
  ggplot(aes(x=nletters)) +
  geom_histogram(binwidth=1) +
  geom_vline(xintercept=mean(nchar(words)),
             colour="green", size=1, alpha=.5) +
  labs(x="Number of Letters", y="Number of Words")

# Build the tf-idf matrix and use it to filter the document term m --------
corpus.dtm <- corpus.dtm[,freqs$tfidf >= 0.01]
corpus.dtm <- corpus.dtm[row_sums(corpus.dtm) > 0,]
term.tfidf.df <- as.data.frame(inspect(corpus.dtm))

# LDA Model - Working ----------------------------------------------------------
# Build a topic model and collect relevant data in data frames.                #
# ---------------------------------------------------------------------------- #
# Problems to install topicmodels package. Solved:
# sudo apt-get install libgsl0-dev

best.model <- lapply(seq(2, 10, by = 1), function(d){LDA(term.tfidf.df, d)}) # this will make a topic model for every number of topics between 2 and 10... it will take some time!
best.model.logLik <- as.data.frame(as.matrix(lapply(best.model, logLik))) # this will produce a list of logLiks for each model...
# plot the distribution of logliklihoods by topic
best.model.logLik.df <- data.frame(topics=c(2:10), LL = as.numeric(as.matrix(best.model.logLik)))

pdf("./plots/1_LogLikelihoodOftheModel.pdf")
ggplot(best.model.logLik.df, aes(x = topics, y = LL))+
  geom_point()+
  xlab("Number of topics") +
  ylab("Log likelihood of the model") +
  geom_line() +
  theme_bw() +
  theme(axis.title.x = element_text(vjust = -0.5, size = 14)) +
  theme(axis.title.y= element_text(size = 14, angle=90, vjust= 0.5)) +
  theme(plot.margin = unit(c(1,1,2,2), "lines"))
dev.off()
# it's not easy to see exactly which topic number has the highest LL, so let's look at the data
best.model.logLik.df.sort <- best.model.logLik.df[order(-best.model.logLik.df$LL), ] # sort to find out which number of topics has the highest loglik, in this case 23 topics.
best.model.logLik.df.sort # have a look to see what's at the top of the list, the one with the highest score
ntop <- best.model.logLik.df.sort[1,]$topics

lda <- LDA(term.tfidf.df, ntop) # generate a LDA model the optimum number of topics
lda.topics <- get_topics(lda, 10) # create object with top 10 topics per document
lda.terms <- get_terms(lda, 10) # get keywords for each topic, just for a quick look. term <- apply(term, MARGIN = 2, paste, collapse = ", ")
beta <- lda@beta # create object containing parameters of the word distribution for each topic
gamma <- lda@gamma # create object containing posterior topic distribution for each document
terms <- lda@terms # create object containing terms (words) that can be used to line up with beta and gamma
colnames(beta) <- terms # puts the terms (or words) as the column names for the topic weights.
id <- t(apply(beta, 1, order)) # order the beta values
beta_ranked <- lapply(1:nrow(id),function(i)beta[i,id[i,]]) # gives table of words per topic with words ranked in order of beta values. Useful for determining the most important words per topic
beta_ranked
gamma
terms
#Extracting the topics per document
#If you want to e.g. correlate topics with sentiment or add the topics as features to the machine learning, it is useful to extract which documents belong to which topic. The fit object contains the needed information, which can be cast into a matrix:

assignments = data.frame(i=lda@wordassignments$i, j=lda@wordassignments$j, v=lda@wordassignments$v)
docsums = acast(assignments, i ~ v, value.var='j', fun.aggregate=length) 
dim(docsums)

topic.labels <- apply(lda.terms, 2, function(x) paste(x, collapse=", "))

gamma.df <- as.data.frame(lda@gamma)
names(gamma.df) <- c(1:ntop)
gamma.df

# Now for each doc, find just the top-ranked topic   
toptopics <- as.data.frame(cbind(document = row.names(gamma.df), 
                                 topic = apply(gamma.df,1,function(x) names(gamma.df)[which(x==max(x))])))
na.omit(toptopics)

# Put output into various csv files
lda_topics <- t(lda.topics)
write.csv(lda_topics, file = "lda10_topics.csv")
write.csv(lda.terms, file = "lda10_terms.csv")

# Output terms with weights, by topic (top 50 terms)
term_list <- lapply(beta_ranked,function(x) exp(x[1:50]))
#term_list <- lapply(term_list,function(x) cbind(names(x),x))
output <- c()
for (i in 1:length(term_list)){
  output <- cbind(output,term_list[[i]])
}
output
write.csv(output, file = "lda10_terms_weight.csv")

m = topmod.lda.fit(filtered_dtm, K = ntop, num.iterations = 1000)
terms(m, 10)[, 1:5] # show first 5 topics, with ten top words per topic
#One of the thing we can do with the LDA topics, is analyze how much attention they get over time, and how much they are used by different sources (e.g., people, newspapers, organizations). To do so, we need to match this article metadata. We can order the metadata to the documents in the LDA model by matching it to the documents slot.

meta = load.data[match(m@documents, load.data$id), ]
#We can now do some plotting. First, we can make a wordcloud for a more fancy (and actually quite informative and intuitive) representation of the top words of a topic.
#topic_term_matrix = posterior(m)$terms
#pdf("./plots/TopWordsOfaTopic.pdf")
#topics.plot.wordcloud(topic_term_matrix, topic_nr = 1)
#dev.off()

#An interesting technique to use on a document-term matrix is that of LDA topic modeling using the r lda package. Topic modeling essentially reduces the dimensionality of the word space by assuming that each document contains a number of (latent) topics, which in turn contain a number of words (Blei et al, JMLR 2003). Before fitting the topic model, it is best to reduce the vocabulary by selecting only informative words. Then, we can fit a topic model using the lda.fit function, which requires a dtm object and the number of topics:
terms.filtered = corpus.dtm[, colnames(corpus.dtm) %in% voca]
m = lda.fit(terms.filtered, K = ntop, num.iterations=1000)
top.topic.words(m$topics)[ ,1:4]
pdf("./plots/TopWordsOfaTopic.pdf")
m$meta = load.data[match(rownames(m$corpus.dtm), load.data$id), ]
lda.plot.wordcloud(m, 10)
dev.off()
#The m object is the standard object returned by the lda package, which also provides functions for inspecting its contents:
top.topic.words(m$topics)

# Predicting the last four document in the corpus (which was left out in the estimation/training)
# Unsupervised classification
posterior_lda <- posterior(lda)
lda_topics <- data.frame(t(posterior_lda$topics))

# Build a topic model and collect relevant data in data frames  -----------
F <- corpus.dtm.rs.m/rowSums(corpus.dtm.rs.m) ## divide by row (doc totals)
classpca <- prcomp(na.omit(F), scale=TRUE)
plot(classpca) 

## look at the big rotations (it does a pretty good job!)
classpca$rotation[order(abs(classpca$rotation[,1]),decreasing=TRUE),1][1:10]
classpca$rotation[order(abs(classpca$rotation[,2]),decreasing=TRUE),2][1:10]

## Plot the first two PCs..
pdf("./plots/1_PCA_distribution.pdf")
plot(classpca$x[,1:2], col=0, xlab="PCA 1 direction", ylab="PCA 2 direction", bty="n")
text(x=classpca$x[,1], y=classpca$x[,2], labels=colnames(corpus.dtm.rs),cex=.4)
dev.off()

## **** a quick topic-modelling example **** ##
## you can give topics a few K, and it chooses the best by BIC
tpc <- topics(corpus.dtm.rs, K=2:ntop) # log(BF) is basically -BIC

## summary prints terms by 'lift': p(term|topic)/p(term)
summary(tpc, ntop) #10 is number of top terms to print

## the topic-term probabilities are called 'theta', and each column is a topic
## we can use these to rank terms by probability within topics
rownames(tpc$theta)[order(tpc$theta[,1], decreasing=TRUE)[1:10]]
rownames(tpc$theta)[order(tpc$theta[,2], decreasing=TRUE)[1:10]]

## plot the papers another way (do them in order)
pdf("./plots/1_TermsandTopics.pdf")
par(srt=-30, xpd=NA) ## rotate stings, and allow words outside plot
plot(tpc$omega[,1], type="l", col=8, xlab="", xlim=c(0.5,12),
     xaxt="n", ylab="topic 1 weight", bty="n")
text(x=1:nrow(tpc$omega), y=tpc$omega[,1], labels=colnames(corpus.dtm.rs),cex=.8)
dev.off()

# Build the incidence matrice and document-topic network ------------------
dt.df <-na.omit(toptopics)
dt.matrix <- as.matrix(table(dt.df))
dt.network <- graph.incidence(dt.matrix)

# set.seed(122)
k <- ntop
corpus.dtm.m <- as.matrix(corpus.dtm.rs)
kmeansResult <- kmeans(corpus.dtm.rs, k)
round(kmeansResult$centers, digits = 3)
cost_df <- data.frame()
for (i in 1:k) {  
  cat(paste("cluster ", i, ": ", sep = ""))  
  s <- sort(kmeansResult$centers[i, ], decreasing = T)  
  kmeans <- kmeans(corpus.dtm.rs.m, centers=i, iter.max=k)
  cat(names(s)[1:5], "\n")
  cost_df<- rbind(cost_df, cbind(i, kmeans$tot.withinss))
}
names(cost_df) <- c("cluster", "cost")

#Calculate lm's for emphasis
lm(cost_df$cost[1:10] ~ cost_df$cluster[1:10])
lm(cost_df$cost[10:19] ~ cost_df$cluster[10:19])

cost_df$fitted <- ifelse(cost_df$cluster <10, (19019.9 - 550.9*cost_df$cluster),
                 ifelse(cost_df$cluster <20, (15251.5 - 116.5*cost_df$cluster),
                 (13246.1 - 35.9*cost_df$cluster)))

#Cost plot
ggplot(data=cost_df, aes(x=cluster, y=cost, group=1)) +
  theme_bw(base_family="Garamond") +
  geom_line(colour = "darkgreen") +
  theme(text = element_text(size=20)) +
  ggtitle("Reduction In Cost For Values of 'k'\n") +
  xlab("\nClusters") +
  ylab("Within-Cluster Sum of Squares\n") +
  scale_x_continuous(breaks=seq(from=0, to=30, by= 5)) +
  geom_line(aes(y= fitted), linetype=2) 
#The plot above is a technique known informally as the ‘elbow method’, where we are looking for breakpoints in our cost plot to understand where we should stop adding clusters. We can see that the slope of the cost function gets flatter at 10 clusters, then flatter again around 20 clusters. This means that as we add clusters above 10 (or 20), each additional cluster becomes less effective at reducing the distance from the each data center (i.e. reduces the variance less). So while we haven’t determined an absolute, single ‘best’ value of ‘k’, we have narrowed down a range of values for ‘k’ to evaluate. Ultimately, the best value of ‘k’ will be determined as a combination of a heuristic method like the ‘Elbow Method’, along with analyst judgement after looking at the results. Once you’ve determined your optimal cluster definitions, it’s trivial to calculate metrics such as Bounce Rate, Pageviews per Visit, Conversion Rate or Average Order Value to see how well the clusters actually describe different behaviors on-site.
pamResult <- pamk(corpus.dtm.rs.m, metric="manhattan")
k <- pamResult$nc
pamResult <- pamResult$pamobject
for (i in 1:k){
  cat("cluster", i, ": ",
      colnames(pamResult$medoids)[which(pamResult$medoids[i,]==1)], "\n")
}
layout(matrix(c(1,2), 1, 2))
plot(pamResult, col.p = pamResult$clustering)
layout(matrix(1))

# SVM - Classifier --------------------------------------------------------
set.seed(123)
train_set <- load.data[sample(nrow(load.data), 494),]
test_set <- tail(train_set, 194)
train_set <- head(train_set, 300)
train_test_set <- rbind(train_set,test_set) 
result <- as.DocumentTermMatrix(cbind(corpus.dtm.rs[,which(colnames(corpus.dtm.rs)%in% colnames(corpus.dtm))],corpus.dtm.rs.m),weighting=weightTfIdf)
train_test_dataframe <- as.data.frame(inspect( result ))
classes <- c(train_test_set$Recommend)
train_df_withoutclass <- head(train_test_dataframe,194)
test_df_withoutclass <- tail(train_test_dataframe,300)
train_classes <- as.factor(head(classes,194))
test_classes <- as.factor(tail(classes,300))

NB <- naiveBayes(train_df_withoutclass,train_classes,laplace = 1, na.rm = TRUE)
PredictionNB <- predict(NB,test_df_withoutclass)
table(PredictionNB,test_classes)
prop.table(table(test_classes==PredictionNB))
summary(NB)

SVM <- svm(train_df_withoutclass,train_classes,kernel="linear")
PredictionSVM <- predict(SVM,test_df_withoutclass)
table(PredictionSVM,test_classes)
prop.table(table(test_classes==PredictionSVM))
summary(SVM)

#knn(train_df_withoutclass[, freqs$term], test_df_withoutclass[, freqs$term], train_classes, k=ntop)
