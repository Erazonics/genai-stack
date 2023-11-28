class BaseLogger:
    def __init__(self) -> None:
        self.info = print


def extract_title_and_question(input_string):
    lines = input_string.strip().split("\n")

    title = ""
    question = ""
    is_question = False  # flag to know if we are inside a "Question" block

    for line in lines:
        if line.startswith("Title:"):
            title = line.split("Title: ", 1)[1].strip()
        elif line.startswith("Question:"):
            question = line.split("Question: ", 1)[1].strip()
            is_question = (
                True  # set the flag to True once we encounter a "Question:" line
            )
        elif is_question:
            # if the line does not start with "Question:" but we are inside a "Question" block,
            # then it is a continuation of the question
            question += "\n" + line.strip()

    return title, question


def create_vector_index(driver, dimension: int) -> None:
    # Adjust the index creation for PDF data
    index_query = ("CALL db.index.vector.createNodeIndex('pdf_storage', 'PdfDocument', 'embedding', $dimension, "
                   "'cosine')")
    try:
        driver.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass


def create_vector_index_pdf(driver, dimension: int) -> None:
    # Adjust the index creation for PDF data
    index_query = ("CALL db.index.vector.createNodeIndex('pdf_storage', 'PdfDocument', 'embedding', $dimension, "
                   "'cosine')")
    try:
        driver.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass


def create_constraints(driver):
    # Adjust the constraints for the PDF data schema
    driver.query(
        "CREATE CONSTRAINT pdf_id IF NOT EXISTS FOR (p:PdfDocument) REQUIRE (p.id) IS UNIQUE"
    )
