import json

from langchain.embeddings import OllamaEmbeddings, OpenAIEmbeddings, SentenceTransformerEmbeddings, BedrockEmbeddings
from langchain.chat_models import ChatOpenAI, ChatOllama, BedrockChat
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import List, Any
from utils import BaseLogger, extract_title_and_question


def load_embedding_model(embedding_model_name: str, logger=BaseLogger(), config={}):
    if embedding_model_name == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=config["ollama_base_url"], model="llama2"
        )
        dimension = 4096
        logger.info("Embedding: Using Ollama")
    elif embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using OpenAI")
    elif embedding_model_name == "aws":
        embeddings = BedrockEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using AWS")
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2", cache_folder="/embedding_model"
        )
        dimension = 384
        logger.info("Embedding: Using SentenceTransformer")
    return embeddings, dimension


def load_llm(llm_name: str, logger=BaseLogger(), config={}):
    if llm_name == "gpt-4":
        logger.info("LLM: Using GPT-4")
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    elif llm_name == "gpt-3.5":
        logger.info("LLM: Using GPT-3.5")
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    elif llm_name == "claudev2":
        logger.info("LLM: ClaudeV2")
        return BedrockChat(
            model_id="anthropic.claude-v2",
            model_kwargs={"temperature": 0.0, "max_tokens_to_sample": 1024},
            streaming=True,
        )
    elif len(llm_name):
        logger.info(f"LLM: Using Ollama: {llm_name}")
        return ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=llm_name,
            streaming=True,
            # seed=2,
            top_k=20,
            # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.5,
            # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more
            # focused text.
            num_ctx=6072,  # Sets the size of the context window used to generate the next token.
        )
    logger.info("LLM: Using GPT-3.5")
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)


def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that helps a support agent with answering programming questions.
    If you don't know the answer, just say that you don't know, you must not make up an answer.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
            user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        chain = prompt | llm
        answer = chain.invoke(
            {"question": user_input}, config={"callbacks": callbacks}
        ).content
        return {"answer": answer}

    return generate_llm_output


def configure_homogenous_materials_chain(llm):
    template = f""" You are an expert in the automotive industry with all the components, semi-components and 
    materials used. Your job is analyzing an IMDS Material Data Sheets. Here's what you need to know:

IMDS (International Material Data System) Context

An MDS (Material Data Sheet) is built up as a tree structure following a hierarchical parent/child relationship. Each 
branching point in the structure is called a node. The higher node is called the “parent” and a node directly 
attached to the parent is called a “child”.

Rule 4.1.A: Child nodes of the same parent node must be of the same type (ex. a semi-component parent node may 
consist of all semi-component child nodes or all material child nodes, but not a mixture of semi-component and 
material child nodes). A mixture of components with semi-components or materials on the same level is allowed, 
if the material or semi-component is not an article, but a coating, lubricant or similar, added to the component.

Components: A component is used to represent a single part, a complete assembly or a complete part within an 
assembly. A complete part on a lower level is usually called a sub-component. A sub-component is described by the 
same symbol as a component.

Rule 4.2.1.A:A component node must have at least one sub-component, one semi-component or one material child node.

Rule 4.2.1.C: The top node component name must be descriptive.

Semi-components: A semi-component is a semi-finished product (example: steel coil, pipe, leather hide, plated steel) 
that will go through further process steps (example: cutting, stamping) to make a finished component. A 
semi-component can contain several materials or semi-components.

Rule 4.3.1.A: A semi-component parent node must have at least one material or one semi-component child node.

Rule 4.3.1.C: The semi-component must be reported in the state which it will have in the finished component. Removal 
ties, wraps, liners etc. must not be reported.

Rule 4.3.1.E: The top node semi-component name (article name) must be descriptive.

Materials: A material normally consists only of basic substances. In some cases a material can consist of other 
materials (example: filled thermoplastics consisting of the materials: basic polymer, master batch colour and master 
batch flame retardant that are processed into a new coloured, flame-retarding, filled thermoplastic compound).

Rule 4.4.1.A: A material parent node must have at least one substance or two material child nodes attached to it.

Rule 4.4.1.B: A material must be described in its end state. Only basic substances contained in the final material 
are to be reported (example: cured adhesives or paint coatings are entered without the evaporating solvents).

Guideline 4.4.1.a: A polymer material should have at least two substances attached to it.

Input: JSON and a report that shows where parent and child nodes are materials inside the JSON.
 
Task: Analyze the following JSON representation of an MDS and the report for it. Identify whether or not the child 
material nodes from the report can mix homogenous to the parent material node. If they can't mix it is not allowed to 
have this representation. To solve that task you need to look at the respective names and types for each node and 
figure out if they could mix in any way or not to make the decision. If however, the name indicates some kind of 
coating or layering in general a non-homogenous product, the child nodes of type material need to be part of a 
component or semi-component parent node. In this case, the homogeneity rule does not apply since the materials do not 
mix.

Homogenous Example: Scenario: Imagine a material node representing an "Aluminum Alloy Frame." This node can have 
child material nodes representing "Aluminum Alloy 6061" and "Aluminum Alloy 7075."

Explanation: Both "Aluminum Alloy 6061" and "Aluminum Alloy 7075" are variations of the base material aluminum alloy. 
They can mix homogeneously because they are made from the same base material (aluminum) but differ in their alloy 
composition and properties. This is allowed because they are variations of a homogeneous base material, 
thus complying with the guidelines that permit different materials as child nodes for a parent material node when 
they mix homogeneously.

Non-Homogenous Example: Scenario: A material node representing a "Plastic Encased Electronic Component." This node 
cannot have a child material node representing "Silicon Semiconductor" and another representing "Polyethylene Plastic 
Casing" as direct children under the same parent material node.

Explanation: The "Silicon Semiconductor" and "Polyethylene Plastic Casing" represent materials that do not mix 
homogeneously; one is a semiconductor material, and the other is a plastic material. This structure implies a 
non-homogenous mix of materials that serve different functions within the electronic component, violating the rule 
against having non-homogenous materials directly under the same parent material node. Instead, the "Silicon 
Semiconductor" should be part of a component or semi-component node that represents the functional electronic part, 
while the "Polyethylene Plastic Casing" could be another component or semi-component node representing the casing. 
This separation respects the rule that materials that do not mix homogeneously need to be part of distinct components 
or semi-components.
 
Expected Output:
List of violating parent material nodes:
Explanation: A brief explanation of why each violation occurs.
Give an confidence score from 1-10 with 1 low and 10 high, indicating how confident you are in your answer. 
        """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{data}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
            user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        if isinstance(user_input, dict):
            data = user_input
        else:
            # Parse the JSON input
            data = json.loads(user_input)

        # List to store nodes that violate Rule 4.4.1.D
        violations = []

        def check_node(node, parent_material_id=None, parent_material_name=None):
            # Check if the node is a material
            if node.get('type') == 'material':
                # If parent is also a material, it violates the rule
                if parent_material_id is not None:
                    violations.append({
                        "parent_id": parent_material_id,
                        "parent_name": parent_material_name,
                        "child_id": node['id'],
                        "child_name": node['name']
                    })

            # If current node is material, pass its id and name as parent for its children
            current_parent_id = node['id'] if node.get('type') == 'material' else parent_material_id
            current_parent_name = node['name'] if node.get('type') == 'material' else parent_material_name

            # Recursively check all child nodes
            for child in node.get('children', []):
                check_node(child, current_parent_id, current_parent_name)

        # Start the traversal at the root of the JSON tree
        check_node(data)

        # Format the violations for output
        violations_str = '; '.join([
            f"Parent Node ID: {violation['parent_id']}, Parent Node Name: {violation['parent_name']}, "
            f"Child ID: {violation['child_id']}, Child Name: {violation['child_name']}"
            for violation in violations
        ])

        # Pass the list of violations to the language model
        extra_context = (
            f"Report: There are {len(violations)} instances where the parent and child nodes are materials. "
            f"These nodes are: {violations_str}")

        chain = prompt | llm
        answer = chain.invoke(
            {"data": user_input, "extra_context": extra_context}, config={"callbacks": callbacks}
        ).content
        return {"answer": answer}

    return generate_llm_output


def configure_fbom_chain(llm):
    template = f"""
    Task Description: Analyze a JSON file that contains information about components and their 
    materials to determine if it has a hierarchical (tree-like) structure. This structure is characterized by nested 
    arrays or objects that represent the relationship between components and their sub-components, down to the 
    material level.
    Data Structure: The JSON file includes components with attributes such as name, part number, 
    weight, and a list of sub-components or materials. Each sub-component or material can have its own properties 
    and, potentially, further nested sub-components. 
    Preferred Structure Criteria: A preferred JSON structure is 
    hierarchical, with components and their materials/sub-components organized in a tree-like fashion. This allows 
    for representing complex relationships and dependencies between different parts of a product.
    Here's a good example of a hierarchical structure:
    
    - Component
  - Subcomponent1
    - Sub-subcomponent1
      - Material1
        - Substance1
        - Substance2
        - Substance3
    - Sub-subcomponent2
      - Material2
        - Substance1
        - Substance2
        - Substance3
  - Subcomponent2
    - Material3
      - Substance1
      - Substance2
      - Substance3
      - Substance4
  - Subcomponent3
    - Sub-subcomponent1
      - Material4
        - Substance1
        - Substance2
    - Sub-subcomponent2
      - Material5
        - Substance1
    - Sub-subcomponent3
      - Material6
        - Substance1
        - Substance2
    
    This structure represents the hierarchy of components and their sub-components, down to the material level. Each 
    indentation level represents a deeper level in the hierarchy.
    
    Here's a bad exmaple of a flat  structure which is not allowed:
    
    - Component
  - Substance1
  - Substance2
  - Substance3
  - Substance1
  - Substance2
  - Substance3
  - Substance4
  - Substance5
  - Substance6
  - Substance7
  - Substance8
  - Substance4
  - Substance7
  - Substance9
  - Substance10
  - Substance11
  - Substance12
  
  This structure is flat, meaning it doesn't show the relationship between components and their sub-components. 
  Instead, all materials are listed at the same level under the main component.
    
    Input: The JSON data will be presented as a text input directly within the environment.
    
    Output: The output should identify whether the JSON structure is hierarchical as desired. If not, 
    provide a brief explanation identifying which part of the structure deviates from the preferred hierarchical 
    organization. Only answer the question based on the data provided, do not make up an answer.
    
    Additional context: A Bill of materials (BOM) is considered to be flat when only the materials of an 
    assembly are listed, but not the subparts. This means that only the assembly-level is given, and not the 
    sub-part-structure with sub-sub-parts down to the smallest possible article. Flat Bom reporting can be a problem 
    because according to EU-regulation "REACH", any presence of a declarable substance above 0.1% in the article hast to 
    be communicated to the authorities and ti your customer. To determine if the substance-% is above 0.1%, VW needs to 
    know what is the smallest article inside an assembly. Based on this information, analyze the following data sheet and 
    identify any violations of the rules provided. If there are any violations, specify which rule(s) have been violated 
    and provide a brief explanation of the violation. If there are no violations, state that the FBOM report is compliant 
    with the rules provided.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
            user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        chain = prompt | llm
        answer = chain.invoke(
            {"question": user_input}, config={"callbacks": callbacks}
        ).content
        return {"answer": answer}

    return generate_llm_output


def configure_rulecheck_chain(llm, neo4j_graph):
    # Check input against certain rules
    records = neo4j_graph.query("MATCH (r:Rule) RETURN r.title AS title, r.body AS body")
    rules = []
    for i, rule in enumerate(records, start=1):
        rules.append((rule["title"], rule["body"]))
    rules_prompt = ""
    for i, rule in enumerate(rules, start=1):
        rules_prompt += f"{i}. \n{rule[0]}\n----\n\n"
        rules_prompt += "----\n\n"

    template = f""" You are an expert in checking rules. Your job is it to check if the following rules apply to the
    given data.
    {rules_prompt} Additionl context: A BOM is considered to be flat when only the materials of an assembly are 
    listed, but not the subparts. This means that only the assembly-level is given, and nit tge sub-part-structure 
    with sub-sub-parts down to the smallest possible article. Flat Bom reporting can be a problem because according to 
    EU-regulation "REACH", any presence of a declarable substance above 0.1% in the article hast to be communicated 
    to the authorities and ti your customer. To determine if the substance-% is above 0.1%, VW needs to know what is 
    the smallest article inside an assembly. Based on this information, analyze the following data sheet and identify 
    any violations of the rules provided. If there are any violations, specify which rule(s) have been violated and 
    provide a brief explanation of the violation. If there are no violations, state that the FBOM report is compliant 
    with the rules provided."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{data}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
            json_data: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        chain = prompt | llm
        answer = chain.invoke(
            {"data": json_data}, config={"callbacks": callbacks}
        ).content
        return {"answer": answer}

    return generate_llm_output


def configure_qa_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    # RAG response for PDF data
    general_system_template = """Use the context from the data you get to answer the question. Make sure to rely on 
    information from the data provided and not make up an answer. If you don't know the answer based on the context, 
    just say that you don't know.
    ---- {summaries} ----
    At the end of each answer include the source of the data you used to answer the question.
    You can find it under the source value.
    Generate accuarate answers with references to the source of the data.
    """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Vector + PDF data response
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database="neo4j",
        index_name="pdf_storage",  # Adjusted to PDF storage index name
        text_node_property="text",  # Assuming text property holds PDF data
        retrieval_query="""
    WITH node AS pdf_chunk, score AS similarity
    RETURN '##PDF Chunk: ' + pdf_chunk.text AS text, similarity as score, {source: 'PDF Data'} AS metadata
    ORDER BY similarity DESC // best matches first
    """,
    )

    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 3}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=3375,
    )
    return kg_qa


def generate_ticket(neo4j_graph, llm_chain, input_question):
    # Get high ranked questions
    records = neo4j_graph.query(
        "MATCH (q:Question) RETURN q.title AS title, q.body AS body ORDER BY q.score DESC LIMIT 3"
    )
    questions = []
    for i, question in enumerate(records, start=1):
        questions.append((question["title"], question["body"]))
    # Ask LLM to generate new question in the same style
    questions_prompt = ""
    for i, question in enumerate(questions, start=1):
        questions_prompt += f"{i}. \n{question[0]}\n----\n\n"
        questions_prompt += f"{question[1][:150]}\n\n"
        questions_prompt += "----\n\n"

    gen_system_template = f"""
    You're an expert in formulating high quality questions. 
    Formulate a question in the same style and tone as the following example questions.
    {questions_prompt}
    ---

    Don't make anything up, only use information in the following question.
    Return a title for the question, and the question post itself.

    Return format template:
    ---
    Title: This is a new title
    Question: This is a new question
    ---
    """
    # we need jinja2 since the questions themselves contain curly braces
    system_prompt = SystemMessagePromptTemplate.from_template(
        gen_system_template, template_format="jinja2"
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            SystemMessagePromptTemplate.from_template(
                """
                Respond in the following template format or you will be unplugged.
                ---
                Title: New title
                Question: New question
                ---
                """
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    llm_response = llm_chain(
        f"Here's the question to rewrite in the expected format: ```{input_question}```",
        [],
        chat_prompt,
    )
    new_title, new_question = extract_title_and_question(llm_response["answer"])
    return (new_title, new_question)
