#!/bin/bash

th ../../summarize.lua fnn_nh0_n0  result/fnn_nh0_nnpl4_d_*[!a-z]_0*.table | grep -v "\t" >  result/summary_n0_comb.csv 
th ../../summarize.lua fnn_nh1_n0  result/fnn_nh1_nnpl4_d_*[!a-z]_0*.table | grep -v "\t" >> result/summary_n0_comb.csv 
th ../../summarize.lua fnn_nh2_n0  result/fnn_nh2_nnpl4_d_*[!a-z]_0*.table | grep -v "\t" >> result/summary_n0_comb.csv 
th ../../summarize.lua grnn_n0     result/grnn_d_*[!a-z]_0*.table          | grep -v "\t" >> result/summary_n0_comb.csv 

th ../../summarize.lua fnn_nh0_n1  result/fnn_nh0_nnpl4_d_*noise*.table  | grep -v "\t" >  result/summary_n1_comb.csv 
th ../../summarize.lua fnn_nh1_n1  result/fnn_nh1_nnpl4_d_*noise*.table  | grep -v "\t" >> result/summary_n1_comb.csv 
th ../../summarize.lua fnn_nh2_n1  result/fnn_nh2_nnpl4_d_*noise*.table  | grep -v "\t" >> result/summary_n1_comb.csv 
th ../../summarize.lua grnn_n1     result/grnn_d_*noise*.table           | grep -v "\t" >> result/summary_n1_comb.csv 
