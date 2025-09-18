const dotenv = require("dotenv").config();
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { ChatMistralAI } = require("@langchain/mistralai");
const { HumanMessage, AIMessage } = require("@langchain/core/messages"); // stm
const { PromptTemplate } = require("@langchain/core/prompts");

// const model = new ChatGoogleGenerativeAI({
//   model: "gemini-2.0-flash",
//   temperature: 0,
//   apiKey: process.env.GOOGLE_API_KEY,
// });
console.clear();

const model = new ChatMistralAI({
  model: "mistral-large-latest",
  temperature: 0,
});

// model.stream("write haiku about the sea").then(async (response) => {
//   for await (const chunk of response) {
//     console.log(chunk.text);
//   }
// });

// const chatHistroy = [
//   new HumanMessage("hey"),
//   new AIMessage("hello"),
//   new HumanMessage("what was my first question?"),
// ];

const template = PromptTemplate.fromTemplate(
  `explain {topic} in a simple way for a {audience} to understand.`
);

// template.invoke({
//     topic:"express",
//     audience:"10 year old"
// }).then(temp =>model.invoke(temp).then((response) => console.log(response.text)))

const chain = template.pipe(model);
chain
  .invoke({
    topic: "express",
    audience: "10 year old",
  })
  .then((response) => console.log(response.text));