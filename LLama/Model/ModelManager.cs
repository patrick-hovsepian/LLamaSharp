using System.Collections.Generic;
using System.Threading.Tasks;

namespace LLama;

internal interface IModelManager
{
    public IEnumerable<string> ModelDirectories { get; }
    public IEnumerable<string> ModelFiles { get; }

    public Task<bool> Load(string modelPath, string alias);
    public Task<bool> Unload(string alias);
    public Task<bool> UnloadAll();

    public IEnumerable<LLamaWeights> GetLoadedModels();

    public IReadOnlyDictionary<string, string> GetModelMetadata(string alias);
}

internal class ModelManager
{
    public Task<bool> LoadModel(string modelName)
    {
        return Task.FromResult(true);
    }

    public Task<bool> UnloadModel(string modelName)
    {
        return Task.FromResult(true);
    }
}
