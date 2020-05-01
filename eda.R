# Packages
library(dplyr)
library(ggplot2)

# Read in data
text <- read.csv('./data/clean_mea_text.csv') # read.csv works better for text
reasons <- read.csv('./data/mea_reasons_filtered.csv')

# Data types
text <- text %>% 
  mutate_all(as.character)
reasons <- reasons %>% 
  mutate_all(as.character)

# Remove labels
toremove <- c('Unknown', 
            'Other', 
            'Did not appear for the Hearing', 
            'Other failure to disclose', 
            'Unknown failure to disclose', 
            'Allowed falsification and misrepresentation of loans',
            'Circumvented the requirements that a branch manager of a licensed mortgage broker have at least three years experience',
            "Did not verify or make a reasonable effort to verify the borrower's information",
            'Employed simultaneously by more than one affiliated mortgage banker or mortgage broker',
            'Engaged in fraud',
            'Failure to disclose charges',
            'Violated NC Securities Act',
            'Withdrew appeal',
            'Unsatisfactory credit')
reasons <- reasons[!reasons$reason %in% toremove,]
  
# Filter for docs in both csvs
bothdocs <- intersect(text$docket_num, reasons$docket_num)
text <- text %>% 
  filter(docket_num %in% bothdocs)
reasons <- reasons %>% 
  filter(docket_num %in% bothdocs)

# How important is multi-label classification?
num_rperdoc <- table(reasons$docket_num)
hist(num_rperdoc)
# Although a majority have only one label, 25 documents have two labels, 
# and 10+ documents have three or more.
# In a dataset this small, we should try multi-label classification.
# Especially since our assumption is that a doc
# can contain multiple violations.

# Is there class imbalance?
num_r <- table(reasons$reason)
num_r <- data.frame(num_r)
ggplot(num_r, aes(x=Var1, y=Freq)) + geom_col()
# Can't see the class names but yes.
