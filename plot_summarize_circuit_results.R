#!/usr/local/bin/Rscript

library(ggplot2)
library(splines)
require(mgcv)

#args <- commandArgs(trailingOnly=TRUE)
#strFilename <- args[1]

app19_feedforward1_n0 <- function() {
  strDataFilename <- "/Users/ameen/mygithub/grnn/app/app19_feedforward1/result/summary_n0_comb.csv"
  strFigureFilename <- paste(strDataFilename, "png", sep=".")
  dfData <- read.csv(strDataFilename, header=FALSE)
  gPlot <- ggplot(dfData, aes(x=V1, y=V2, color=V3)) +
    geom_point(shape=20, size=1, position="dodge") +
    coord_cartesian(ylim = c(0, 2.5))+
    xlab("data size") + ylab("MSE")+
    geom_smooth(method = "glm", se=FALSE, method.args= list(family="quasipoisson"), formula = y ~ ns(x, 2))
  ggsave(strFigureFilename, gPlot, width=13.4, height = 6.89)
}

app19_feedforward1_n1 <- function() {
  strDataFilename <- "/Users/ameen/mygithub/grnn/app/app19_feedforward1/result/summary_n1_comb.csv"
  strFigureFilename <- paste(strDataFilename, "png", sep=".")
  dfData <- read.csv(strDataFilename, header=FALSE)
  gPlot <- ggplot(dfData, aes(x=V1, y=V2, color=V3)) +
    geom_point(shape=20, size=1, position="dodge") +
    coord_cartesian(ylim = c(0, 2.5))+
    xlab("data size") + ylab("MSE")+
    geom_smooth(method = "glm", se=FALSE, method.args= list(family="quasipoisson"), formula = y ~ ns(x, 2))
  ggsave(strFigureFilename, gPlot, width=13.4, height = 6.89)
}

app18_net9s_n0 <- function(){
  strDataFilename <- "/Users/ameen/mygithub/grnn/app/app18_net9s/result/summary_n0_comb.csv"
  strFigureFilename <- paste(strDataFilename, "png", sep=".")
  dfData <- read.csv(strDataFilename, header=FALSE)
  gPlot <- ggplot(dfData, aes(x=V1, y=V2, color=V3)) +
    geom_point(shape=20, size=1, position="dodge") +
    xlim(c(9,150))+
    #ylim(c(0,5))+
    coord_cartesian(ylim = c(0, 3))+
    xlab("data size") + ylab("MSE")+
    geom_smooth(method = "glm", se=FALSE, method.args= list(family="quasipoisson"), formula = y ~ ns(x, 2))
  ggsave(strFigureFilename, gPlot, width=13.4, height = 6.89)
  print(gPlot)
}

app18_net9s_n1 <- function(){
  strDataFilename <- "/Users/ameen/mygithub/grnn/app/app18_net9s/result/summary_n1_comb.csv"
  strFigureFilename <- paste(strDataFilename, "png", sep=".")
  dfData <- read.csv(strDataFilename, header=FALSE)
  gPlot <- ggplot(dfData, aes(x=V1, y=V2, color=V3)) +
    geom_point(shape=20, size=1, position="dodge") +
    xlim(c(9,90))+
    #ylim(c(0,4))+
    coord_cartesian(ylim = c(0, 3))+
    xlab("data size") + ylab("MSE")+
    geom_smooth(method = "glm", se=FALSE, method.args= list(family="quasipoisson"), formula = y ~ ns(x, 2))
  ggsave(strFigureFilename, gPlot, width=13.4, height = 6.89)
  print(gPlot)
}

app19_feedforward1_n0()
app19_feedforward1_n1()
app18_net9s_n0()
app18_net9s_n1()
