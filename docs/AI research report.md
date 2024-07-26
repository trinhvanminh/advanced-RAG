# AZURE HEALTH BOT

**Azure Health Bot** is a cloud platform that empowers healthcare organizations to build and deploy intelligent and compliant virtual health assistants. It combines natural language understanding, machine learning, and conversational AI to create powerful health chatbots.

## What can it do ?

**Azure Health Bot scenario** provides a variety of [built-in scenarios](https://learn.microsoft.com/en-us/azure/health-bot/bot_docs/help) to enhance its capabilities using these language models:

1.  **Medical information requests**: This model identifies requests for information about different types of medical concepts. For example, 'Tell me about meningitis' or 'What are the symptoms of meningitis' will produce a high confidence score with this model. Depending on the information type requests the respective intent will be produced.
2.  **Medical complaints**: This model identifies medical complaints such as 'My daughter has had a fever for the last three days'. If possible the model extracts important information from the initial utterance such as gender, age, duration of symptoms and the subject of the complaint.
3.  **Drugs and medications**: This model identifies requests for information abouts different types and brands of drugs and medications.

See more [language models](https://learn.microsoft.com/en-us/azure/health-bot/language_models)

## Triage provider

Before building a triage scenario in Azure AI Health Bot, you should set your preferred triage provider. The built-in triage and symptom checking protocols are provided by third parties and licensed by Microsoft

![](https://i.imgur.com/cXjaMFu.png)

Each provider offers different features and uses different clinical methodologies to triage and assess symptoms. The Azure AI Health Bot includes triage engines from multiple providers so that you can select the engine that meets your organizations clinical and functional requirements. The engines currently available are:

- [Capita Healthcare Decisions](https://capitahealthcaredecisions.com/products-and-services/clinical-support/)
- [Infermedica](https://infermedica.com/)

## Demo images

### User asking for help

<img src="https://i.imgur.com/OrhUhjU.png" width="400"/>

https://i.imgur.com/OrhUhjU.png

### Handoff to Agent

<img src="https://i.imgur.com/s7tnnbS.png" width="800"/>

https://i.imgur.com/s7tnnbS.png

## Integration

- **Data Connections**: To integrate with external data resources, we provide a data connection object, which allows you to make HTTPS calls to third-party API providers or your own API endpoints (_General Endpoint_, [_FHIR Endpoint_](https://learn.microsoft.com/en-us/azure/healthcare-apis/azure-api-for-fhir/overview), ..).
  ![](https://i.imgur.com/u73zGZW.png)
- **Channel**: A channel is a connection between the Azure AI Health Bot and communication apps (_Web app, DirectLine, Microsoft Teams_). [See more](https://learn.microsoft.com/en-us/azure/health-bot/channels/main#setting-up-a-channel)

## Demo

> Have some trouble with handoff to live agent feature. It have not available now

- User: [healthcare-bot-agfr4focaqr62.azurewebsites.net](https://healthcare-bot-agfr4focaqr62.azurewebsites.net/)

- Agent (_login with blank credential_): [Health Bot (healthcare-bot-agfr4focaqr62.azurewebsites.net)](https://healthcare-bot-agfr4focaqr62.azurewebsites.net/agent.html)

## Pricing

1.  **Free Tier:**
    - Includes 3,000 messages (up to 10 messages per second).
    - 200 MCUs (Medical Content Consumption Unit).
2.  **Standard Tier ($500/month):**
    - Includes 10,000 messages (up to 100 messages per second).
    - 1,000 MCUs.
    - Overage charges apply: $2.50 per 1,000 messages and $0.18 per MCU

[See more](https://azure.microsoft.com/en-us/pricing/details/bot-services/health-bot/)

# AZURE AI STUDIO

**Azure AI Studio** is a trusted and inclusive platform that empowers developers of all abilities and preferences to innovate with AI and shape the future.

> Some features/models are currently in preview, so they may not be stable yet.

## What can it do ?

### Prompt flow

Prompt Flow is a development tool aimed at enhancing the development cycle of AI applications powered by Large Language Models (LLMs). It offers a complete solution that eases the process of prototyping, experimenting, iterating, and deploying AI applications.

Here's what Prompt Flow can do:

1. _Conversation Management_: Enables developers to craft and manage conversational flows for seamless user-application interactions.
2. _Natural Language Understanding (NLU)_: Azure AI Prompt Flow comprehends natural language inputs from users, facilitating more natural interactions.
3. _Intent Recognition_: Identifies the intent behind user inputs, allowing the system to direct the conversation and provide pertinent responses.
4. _Context Management_: Azure AI Prompt Flow retains context throughout a conversation, understanding references and ensuring coherence across multiple interactions.
5. _Multi-turn Dialogues_: Accommodates multi-turn dialogues, enabling ongoing exchanges between the user and the application to reach the intended result.
6. _Integration with Other Azure Services_: Integrates with various Azure services, including Azure Bot Service, Azure Cognitive Services (Speech and Language APIs), Azure Functions, etc., to broaden its capabilities.
7. _Customization and Extensibility_: Offers developers the ability to tailor and expand Azure AI Prompt Flow's functionality to meet specific application needs, including the integration of custom logic and external systems.
8. _Analytics and Insights_: Provides analytics and insights on user interactions, helping developers to understand user behavior and refine the conversational experience.

_A sample prompt flow of searching by wikipedia._
![](https://i.imgur.com/gKubfTQ.png)

![](https://i.imgur.com/ZDcVywF.png)

### Chat playground

It can understanding prompt structure. GPT-35-Turbo was trained to use special tokens to delineate different parts of the prompt.

Deploy your model or data directly from the Studio and adjust your assistant setup to improve the assistant's responses then you can start chatting with bot assistant.

See the **Demo** sections below.

## How to custom with your own data ?

You can upload your file or folder by click button **Add a new data source**

![](https://i.imgur.com/cxJIYuC.png)

Then you choose 1 of 5 search types here depend on your purpose
![](https://i.imgur.com/xtrBZWU.png)

Search types
![link](https://i.imgur.com/vVdADQr.png)

[See more](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/use-your-data?tabs=ai-search#search-types)

### Demo

Using this custom data: [hospital-test-data.xlsx](https://mylogix-my.sharepoint.com/:x:/g/personal/khoi_tvm_logixtek_com/ERIMJi9f9npGtLcWK06XRasB6987GSq8Hq9LPUM9y1St9A?e=tBhhqq)

After data was indexed, you will get this result
![](https://i.imgur.com/AfG1ueD.png)

After deployment
![](https://i.imgur.com/WyIAAs9.png)

## Pricing

When deploying solutions, **Azure AI services**, **Azure Machine Learning**, and other Azure resources used inside of Azure AI Studio will be billed at their existing [rates](https://azure.microsoft.com/en-us/pricing/details/ai-studio/#pricing).

The cost varies depending on the service you use, but it is typically driven by the deployment services and the model itself.

For OpenAI Services
![](https://i.imgur.com/ZckTmMU.png)
[See more](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)

# Google AI Studio (additional)

A browser-based IDE for prototyping with generative models. Google AI Studio lets you quickly try out models and experiment with different prompts. When you've built something you're happy with, you can export it to code in your preferred programming language and use the [Gemini API](https://ai.google.dev/gemini-api/docs/quickstart) with it.

## Pricing

1. **Gemini 1.5 Flash (_Preview_)**

   **Free of charge**

   - Rate limit
     - 15 RPM (requests per minute)
     - 1 million TPM (tokens per minute)
     - 1,500 RPD (requests per day)
   - Data sharing

   **Pay-as-you-go**

   - Rate limit
     - 360 RPM (requests per minute)
     - 10 million TPM (tokens per minute)
     - 10,000 RPD (requests per day)
   - Price (input)
     - $0.35 / 1 million tokens (for prompts up to 128K tokens)
     - $0.70 / 1 million tokens (for prompts longer than 128K)
   - Price (output)
     - $1.05 / 1 million tokens (for prompts up to 128K tokens)
     - $2.10 / 1 million tokens (for prompts longer than 128K)
   - Protected data

[Other models pricing](https://ai.google.dev/pricing)

## Demo

- Company documents chat bot
![](https://i.imgur.com/zrwYHlK.png)

- [Order system using Gemini AI and Streamlit](https://che-hoa.streamlit.app/)
![](https://i.imgur.com/XnqDlc5.png)

