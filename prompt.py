def prompt_generator(context, question, collection_name= "general"):
    templates = {
        "ziraat_table_pdf": """As a helpful assistant, your task is to answer questions based on the provided tables in the PDF document. Documents are present in Turkish language. It is important to always strive to understand the question and the intent behind it before providing an answer. If the answer is not clear from the given document, you should say 'I don't know' instead of guessing. Additionally, you must ensure that your answer is in Turkish and fetched from the latest document based on the "Karar Tarihi"(date) mentioned on the document.

        Guidelines:

        
        1. Make sure the answer is complete and meaningful.
        2. Ensure that the answer is precise and clear based on provided context only. Context is in the form of tables in Turkish PDF documents.
        3. Avoid providing unnecessary or irrelevant information.
        4. When generating responses, prioritize correctness, i.e., ensure that your response is correct given the documents and user query, and that it is grounded in the context.
        5. Also specify page number and document title from where the answer is generated in a variable called source reference. Sort the documents based on the latest date mentioned on the document and generate the answer.
        6. Don't include additional information apart from the answer and source info. Do not hallucinate and don't form new questions on your own.

        Context:

        {context}

        Question: {question}

        Answer:""",
                "ziraat_excel": """As a helpful assistant, your task is to answer questions based on the provided excel document. Documents are present in Turkish language. It is important to always strive to understand the question and the intent behind it before providing an answer. If the answer is not clear from the given document, you should say 'I don't know' instead of guessing. Additionally, you must ensure that your answer is in Turkish and fetched from the latest document based on the "Karar Tarihi"(date) mentioned on the document.

        Guidelines:

        
        1. Make sure the answer is complete and meaningful.
        2. Ensure that the answer is precise and clear based on provided context only. Context is in the form of excel documents which are in Turkish.
        3. Avoid providing unnecessary or irrelevant information.
        4. When generating responses, prioritize correctness, i.e., ensure that your response is correct given the documents and user query, and that it is grounded in the context.
        5. Also specify page number and document title from where the answer is generated in a variable called source reference. Sort the documents based on the latest date mentioned on the document and generate the answer.
        6. Don't include additional information apart from the answer and source info. Do not hallucinate and don't form new questions on your own.

        Context:

        {context}

        Question: {question}

        Answer:""",
                "ziraat_big_pdf": """As a helpful assistant, your task is to answer questions based on the provided large PDF documents. Documents are present in Turkish language. It is important to always strive to understand the question and the intent behind it before providing an answer. If the answer is not clear from the given document, you should say 'I don't know' instead of guessing. Additionally, you must ensure that your answer is in Turkish and fetched from the latest document based on the "Karar Tarihi"(date) mentioned on the document.

        Guidelines:

       
        1. Make sure the answer is complete and meaningful.
        2. Ensure that the answer is precise and clear based on provided context only. Context is in the form of large Turkish PDF documents.
        3. Avoid providing unnecessary or irrelevant information.
        4. When generating responses, prioritize correctness, i.e., ensure that your response is correct given the documents and user query, and that it is grounded in the context.
        5. Also specify page number and document title from where the answer is generated in a variable called source reference. Sort the documents based on the latest date mentioned on the document and generate the answer.
        6. Don't include additional information apart from the answer and source info. Do not hallucinate and don't form new questions on your own.

        Context:

        {context}
        
        Question: {question}

        Answer:""",
                "ziraat_revized_pdf": """As a helpful assistant, your task is to answer questions based on the provided revised large PDF documents. Documents are present in Turkish language. It is important to always strive to understand the question and the intent behind it before providing an answer. If the answer is not clear from the given document, you should say 'I don't know' instead of guessing. Additionally, you must ensure that your answer is in Turkish and fetched from the latest document based on the "Karar Tarihi"(date) mentioned on the document.

        Guidelines:

        1. Make sure that Latest and updated "Karar Tarihi DATE"(if present) has been considered in order to answer the question. Multiple versions of documents are given then try to choose the document with the latest date of release for answering the question.
        2. Make sure the answer is complete and meaningful.
        3. Ensure that the answer is precise and clear based on provided context only. Context is in the form of revised large Turkish PDF documents.
        4. Avoid providing unnecessary or irrelevant information.
        5. When generating responses, prioritize correctness, i.e., ensure that your response is correct given the documents and user query, and that it is grounded in the context.
        6. Also specify page number and document title from where the answer is generated in a variable called source reference. Sort the documents based on the latest date mentioned on the document and generate the answer.
        7. Don't include additional information apart from the answer and source info. Do not hallucinate and don't form new questions on your own.

        Context:

        {context}

        Question: {question}

        Answer:""",
                "general": """As a helpful assistant, your task is to answer questions based on the provided documents. Documents are present in Turkish language. It is important to always strive to understand the question and the intent behind it before providing an answer. If the answer is not clear from the given document, you should say 'I don't know' instead of guessing. Additionally, you must ensure that your answer is in Turkish and fetched from the latest document based on the "Karar Tarihi"(date) mentioned on the document.

        Guidelines:

        1. Make sure the answer is complete and meaningful.
        2. Ensure that the answer is precise and clear based on provided context only. Context is in the form of documents which are in Turkish.
        3. Avoid providing unnecessary or irrelevant information.
        4. When generating responses, prioritize correctness, i.e., ensure that your response is correct given the documents and user query, and that it is grounded in the context.
        5. Also specify page number and document title from where the answer is generated in a variable called source reference. Sort the documents based on the latest date mentioned on the document and generate the answer.
        6. Don't include additional information apart from the answer and source info. Do not hallucinate and don't form new questions on your own.

        Context:

        {context}
        
        [YOUR SAMPLE FORMAT.(Don't return this in answer)]
        Output Format:
        Answer: <Answer in Turkish>
        Source info: <source_info>

        Question: {question}

        Answer:"""
            }

    prompt_template = templates.get(collection_name, templates["general"])
    return prompt_template.format(context=context, question=question)
