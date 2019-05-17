#Load necessary packages,models, and datasets
library(shiny)
library(data.table)
library(ggplot2)
library(keras)
library(nnet)
library(reticulate)
library(dplyr)
library(glmnet)
library(shinythemes)
library(shinycssloaders)
load("LogisticRegression.rda")
model<-load_model_hdf5("model.hdf5")
z.train<-fread("z.train.csv")
z.test<-fread("z.test.csv")
ftrain<-fread("df_train.csv")
ftest<-fread("df_test.csv")
los.train<-as.data.frame(unclass(fread("los.train.csv")[,-1]))
los.test<-as.data.frame(unclass(fread("los.test.csv")[,-1]))
load("mod.los.RData")
text <- ftrain$TEXT
#Increase size of R app, establish sequence length and dictionary size
options(shiny.maxRequestSize=100*1024^2)
max_features <- 12840
maxlen=200
#fit tokenizer to training data
tokenizer <- text_tokenizer(num_words = max_features)
tokenizer %>% 
    fit_text_tokenizer(text)
#Create ui and server sides. UI contains text and file inputs, including tab and label layout and design
ui <- fluidPage(theme = shinytheme("darkly"),
    titlePanel("Medical File Upload"),
    tabsetPanel(
        tabPanel("Mortality",
    sidebarLayout(
        sidebarPanel(
        fileInput("file1","Choose CSV File",
                  multiple=TRUE,
                  accept=c("text/csv","text/comma-separated-values,text/plain",".csv")),
    tags$hr(),
    checkboxInput("example1","Example Prediction",TRUE),
    numericInput("val1","Example Number",1,nrow(z.test),1),
    checkboxInput("header1","Header",TRUE),
    radioButtons("sep1","Separator",choices=c(Comma = ",",
                                             Semicolon = ";",
                                             Tab = "\t"),
                                    selected = ","),
    radioButtons("quote1", "Quote",
                 choices = c(None = "",
                             "Double Quote" = '"',
                             "Single Quote" = "'"),
                 selected = '"'),
    
    tags$hr()),
    mainPanel("Mortality Prediction",plotOutput("preds1"),tableOutput("tab1")
              )
    )
    ),
    tabPanel("Length Of Stay",
             sidebarLayout(
                 sidebarPanel(
                     fileInput("file2","Choose CSV File",
                               multiple=TRUE,
                               accept=c("text/csv","text/comma-separated-values,text/plain",".csv")),
                     tags$hr(),
                     checkboxInput("example2","Example Prediction",TRUE),
                     numericInput("val2","Example Number",1,nrow(los.test),1),
                     checkboxInput("header2","Header",TRUE),
                     radioButtons("sep2","Separator",choices=c(Comma = ",",
                                                              Semicolon = ";",
                                                              Tab = "\t"),
                                  selected = ","),
                     radioButtons("quote2", "Quote",
                                  choices = c(None = "",
                                              "Double Quote" = '"',
                                              "Single Quote" = "'"),
                                  selected = '"'),
                     
                     tags$hr()),
                 mainPanel("Length of Stay Prediction",plotOutput("preds2"),tableOutput("tab2")
                 )
             )
             ),
    tabPanel("Readmission",
             sidebarLayout(
                 sidebarPanel(
                     fileInput("file3","Choose CSV File",
                               multiple=TRUE,
                               accept=c("text/csv","text/comma-separated-values,text/plain",".csv")),
                     tags$hr(),
                     checkboxInput("example3","Example Prediction",TRUE),
                     numericInput(inputId = "val3",label = "Example Number",1,nrow(ftest),1),
                     checkboxInput("header3","Header",TRUE),
                     radioButtons("sep3","Separator",choices=c(Comma = ",",
                                                              Semicolon = ";",
                                                              Tab = "\t"),
                                  selected = ","),
                     radioButtons("quote3", "Quote",
                                  choices = c(None = "",
                                              "Double Quote" = '"',
                                              "Single Quote" = "'"),
                                  selected = '"'),
                     
                     tags$hr()),
                 mainPanel("Readmission Prediction",plotOutput("preds3") %>% withSpinner(color="#0dc5c1"),
                           tableOutput("tab3")
                 )
             )
    )
    )
)
#create server side. The reactive output includes histograms with a fitted vertical prediction line
server <- function(input, output) {
    output$preds1 <- renderPlot({
        if(input$example1){
            df <- z.test
            i<-predict(model2,df,type="response")[input$val1]
            qplot(z.train$VALHOSPITAL_EXPIRE_FLAG,geom="histogram",binwidth=.40)+
                geom_vline(xintercept = i,colour="#0dc5c1",size=3)+
                xlab("Mortality Probability")+
                geom_text(aes(x=i+.1, label=round(i,digits=3), y=10000), colour="#0dc5c1", angle=30, text=element_text(size=20))+
                theme(axis.title.x=element_blank(),
                      axis.text.x=element_blank(),
                      axis.ticks.x=element_blank())
            
        
        }
        else{ 
        df <- read.csv(input$file1$datapath,
                       header = input$header1,
                       sep = input$sep1,
                       quote = input$quote1)
        i<-predict(model2,df,type="response")[1]
        qplot(z.train$VALHOSPITAL_EXPIRE_FLAG,geom="histogram",binwidth=.40)+
        geom_vline(xintercept = i,colour="#0dc5c1",size=3)+
        xlab("Mortality Probability")+
        geom_text(aes(x=i+.1, label=round(i,digits=3), y=10000), colour="#0dc5c1", angle=30, text=element_text(size=20))+
            theme(axis.title.x=element_blank(),
                  axis.text.x=element_blank(),
                  axis.ticks.x=element_blank())
        
}
    })
    output$tab1<-renderTable({
        if(input$example1){
            df <- z.test[input$val1,c(3,5:6)]
            df
        }
        else{
            df <- read.csv(input$file1$datapath,
                           header = input$header1,
                           sep = input$sep1,
                           quote = input$quote1)
            head(df[1,c(4,6:7)])
        }
    })
    output$preds3 <- renderPlot({
        if(input$example3){
            df <- ftest
            test<-df$TEXT
            test_seqs<-texts_to_sequences(tokenizer,test)
            x_test <- test_seqs %>%
                pad_sequences(maxlen = maxlen)
            i<-predict(model,x_test,type="response")[input$val3]
            qplot(ftrain$OUTPUT_LABEL,geom="histogram",binwidth=.40)+
                geom_vline(xintercept = i,colour="#0dc5c1",size=3)+
                xlab("Readmission Probability")+
                geom_text(aes(x=i+.1, label=round(i,digits=3), y=2500), colour="#0dc5c1", angle=30, text=element_text(size=20))+
                theme(axis.title.x=element_blank(),
                      axis.text.x=element_blank(),
                      axis.ticks.x=element_blank())
            
            
        }
        else{ 
            df <- read.csv(input$file3$datapath,
                           header = input$header3,
                           sep = input$sep3,
                           quote = input$quote3)
            test<-df$TEXT
            test_seqs<-texts_to_sequences(tokenizer,test)
            x_test <- test_seqs %>%
                pad_sequences(maxlen = maxlen)
            i<-predict(model,x_test,type="response")[1]
            qplot(ftrain$OUTPUT_LABEL,geom="histogram",binwidth=.40)+
                geom_vline(xintercept = i,colour="#0dc5c1",size=3)+
                xlab("Readmission Probability")+
                geom_text(aes(x=i+.1, label=round(i,digits=3), y=2500), colour="#0dc5c1", angle=30, text=element_text(size=20))+
                theme(axis.title.x=element_blank(),
                      axis.text.x=element_blank(),
                      axis.ticks.x=element_blank())
            
        }
    })
    output$tab3<-renderTable({
        if(input$example3){
            df <- ftest[input$val3,]$TEXT
            df
        }
        else{
            df <- read.csv(input$file3$datapath,
                           header = input$header3,
                           sep = input$sep3,
                           quote = input$quote3)
            (df[1,]$TEXT)
        }
    })
    
    output$preds2 <- renderPlot({
        if(input$example2){
            df <- los.test
            i<-predict(los.mod,df)[input$val2]
            qplot(los.train$LOS,geom="histogram")+
                geom_vline(xintercept = i,colour="#0dc5c1",size=3)+
                xlab("Length of Stay Prediction")+
                geom_text(aes(x=i+10, label=round(i,digits=3), y=2500), colour="#0dc5c1", angle=30, text=element_text(size=20))
            
        }
        else{ 
            df <- read.csv(input$file2$datapath,
                           header = input$header2,
                           sep = input$sep2,
                           quote = input$quote2)
            i<-predict(los.mod,los.test)[1]
            qplot(los.train$LOS,geom="histogram")+
                geom_vline(xintercept = i,colour="#0dc5c1",size=3)+
                xlab("Length of Stay Prediction")+
                geom_text(aes(x=i+10, label=round(i,digits=3), y=2500), colour="#0dc5c1", angle=30, text=element_text(size=20))
        }
    })
    output$tab2<-renderTable({
        if(input$example2){
            df <- los.test[input$val2,1:7]
            df
        }
        else{
            df <- read.csv(input$file2$datapath,
                           header = input$header2,
                           sep = input$sep2,
                           quote = input$quote2)
            head(df[1,2:8])
        }
    })
    
}

shinyApp(ui,server)
