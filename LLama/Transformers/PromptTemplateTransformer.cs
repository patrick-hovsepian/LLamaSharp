using System.Text;
using LLama.Abstractions;
using LLama.Common;

namespace LLama.Transformers;

///
public class PromptTemplateTransformer(LLamaWeights model, 
    bool withAssistant = true) : IHistoryTransform
{
    private readonly LLamaWeights _model = model;
    private readonly bool _withAssistant = withAssistant;

    ///
    public string HistoryToText(ChatHistory history)
    {
        // TODO: cache on creation
        // TODO: maybe use the tokenizer.chat_template and do it natively
        var template = new LLamaTemplate(_model.NativeHandle)
        {
            AddAssistant = _withAssistant,
        };

        if (history.Messages.Count == 1)
        {
            return EncodeMessage(history.Messages[0], template);
        }

        // encode each message and return the final prompt
        StringBuilder sb = new();
        foreach (var message in history.Messages)
        {
            sb.Append(EncodeMessage(message, template));
        }
        return sb.ToString();
    }

    private string EncodeMessage(ChatHistory.Message message, LLamaTemplate template)
    {
        // case sensitive role claim
        template.Add(message.AuthorRole.ToString().ToLowerInvariant(), message.Content);

        // decode and return
        var formattedPrompt = template.ToModelPrompt();

        // add debug printings
        return formattedPrompt;
    }

    ///
    public ChatHistory TextToHistory(AuthorRole role, string text)
    {
        return new ChatHistory([new ChatHistory.Message(role, text)]);
    }

    ///
    public IHistoryTransform Clone()
    {
        // need to preserve history?
        return new PromptTemplateTransformer(_model);
    }
}
