﻿using LLama.Common;
using LLama.Transformers;

namespace LLama.Examples.Examples;

public class LLama3ChatSession
{
    public static async Task Run()
    {
        string modelPath = UserSettings.GetModelPath();

        var parameters = new ModelParams(modelPath)
        {
            Seed = 1337,
            GpuLayerCount = 10
        };
        using var model = LLamaWeights.LoadFromFile(parameters);
        using var context = model.CreateContext(parameters);
        var executor = new InteractiveExecutor(context);

        var chatHistoryJson = File.ReadAllText("Assets/chat-with-bob.json");
        var chatHistory = ChatHistory.FromJson(chatHistoryJson) ?? new ChatHistory();

        ChatSession session = new(executor, chatHistory);

        // add the default templator. If llama.cpp doesn't support the template by default, 
        // you'll need to write your own transformer to format the prompt correctly
        session.WithHistoryTransform(new PromptTemplateTransformer(model, true)); 

        // Add a transformer to eliminate printing the end of turn tokens, llama 3 specifically has an odd LF that gets printed somtimes
        session.WithOutputTransform(new LLamaTransforms.KeywordTextOutputStreamTransform(
            [model.Tokens.EndOfTurnToken!, "�"],
            redundancyLength: 5));

        var inferenceParams = new InferenceParams()
        {
            MaxTokens = -1, // keep generating tokens until the anti prompt is encountered
            Temperature = 0.6f,
            AntiPrompts = [model.Tokens.EndOfTurnToken!] // model specific end of turn string
        };

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("The chat session has started.");

        // show the prompt
        Console.ForegroundColor = ConsoleColor.Green;
        Console.Write("User> ");
        var userInput = Console.ReadLine() ?? "";

        while (userInput != "exit")
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("Assistant> ");

            // as each token (partial or whole word is streamed back) print it to the console, stream to web client, etc
            await foreach (
                var text
                in session.ChatAsync(
                    new ChatHistory.Message(AuthorRole.User, userInput),
                    inferenceParams))
            {
                Console.ForegroundColor = ConsoleColor.White;
                Console.Write(text);
            }
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write("User> ");
            userInput = Console.ReadLine() ?? "";
        }
    }
}
