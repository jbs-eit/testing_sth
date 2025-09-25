# app.R
library(shiny)
library(jsonlite)
library(ggplot2)
library(DT)

# Load the manifest once at app start
manifest <- fromJSON("index.json", simplifyVector = TRUE)

ui <- fluidPage(
  tags$head(tags$link(rel="stylesheet", type="text/css", href="styles.css")),
  titlePanel("Simulation Results (Proof of Concept)"),
  sidebarLayout(
    sidebarPanel(
      selectInput("age", "Age:", choices = manifest$age_values, selected = 65),
      selectInput("scenario", "Scenario:", choices = manifest$scenario_values, selected = "baseline"),
      actionButton("load", "Load results"),
      hr(),
      downloadButton("download_csv", "Download CSV")
    ),
    mainPanel(
      h4(textOutput("caption")),
      plotOutput("plot", height = "320px"),
      DTOutput("table")
    )
  )
)

server <- function(input, output, session) {
  
  # Resolve the right file for the chosen parameters
  resolve_path <- reactive({
    key <- paste(input$age, input$scenario, sep="|")
    rel <- manifest$lookup[[key]]
    shiny::validate(shiny::need(!is.null(rel), "No data for that combination"))
    rel
  })
  
  # Load data when user clicks "Load results"
  data_reactive <- eventReactive(input$load, {
    path <- resolve_path()  # e.g., "data/results_age-40_scn-policy.csv"
    
    # 1) Classic Shiny: read from filesystem under www/
    local_path <- file.path("www", path)
    if (file.exists(local_path)) {
      return(read.csv(local_path, stringsAsFactors = FALSE))
    }
    
    # 2) If you ever keep data at ./data directly
    if (file.exists(path)) {
      return(read.csv(path, stringsAsFactors = FALSE))
    }
    
    # 3) Shinylive/webR: fetch URL relative to app origin
    tmp <- tempfile(fileext = ".csv")
    download.file(path, tmp, mode = "wb", quiet = TRUE)
    read.csv(tmp, stringsAsFactors = FALSE)
  }, ignoreInit = TRUE)
  
  output$caption <- renderText({
    paste0("Age ", input$age, " — Scenario: ", input$scenario)
  })
  
  output$plot <- renderPlot({
    df <- data_reactive()
    ggplot(df, aes(year, metric)) +
      geom_line() +
      labs(x = NULL, y = "Metric (units)", title = unique(df$group)) +
      theme_minimal(base_size = 12)
  })
  
  output$table <- renderDT({
    datatable(data_reactive(), options = list(pageLength = 5), rownames = FALSE)
  })
  
  output$download_csv <- downloadHandler(
    filename = function() {
      key <- paste(input$age, input$scenario, sep="|")
      paste0("results_", gsub("\\|", "_", key), ".csv")
    },
    content = function(file) {
      df <- data_reactive()
      write.csv(df, file, row.names = FALSE)
    }
  )
}

shinyApp(ui, server)