using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using LLama.Common;

namespace LLama.Model;

internal class ModelManagerConfiguration
{
    public IEnumerable<string> ModelDirectories { get; init; } = [];
}

internal interface IModelManager
{
    public IEnumerable<string> ModelDirectories { get; set; }
    public IEnumerable<string> ModelFiles { get; }

    public Task<LLamaWeights> Load(string modelPath,
        Action<ModelParams>? modelConfigurator = null!,
        string modelId = "",
        CancellationToken cancellationToken = default);
    public bool Unload(string modelId);
    public void UnloadAll();

    public bool TryGetLoadedModel(string modeId, out LLamaWeights model);
    public IEnumerable<LLamaWeights> GetLoadedModels();
}

internal class ModelManager : IModelManager
{
    public static string[] ExpectedModelFileTypes = [
        ".gguf"
    ];

    private readonly HashSet<string> _modelDirectories = [];
    private readonly List<string> _modelFiles = [];

    private readonly Dictionary<string, LLamaWeights> _modelCache = [];

    public ModelManager(string[] directories)
    {
        FindAndAddModels(directories);
    }

    private void FindAndAddModels(IEnumerable<string> dirs)
    {
        foreach (var dir in dirs)
        {
            if (!Directory.Exists(dir))
            {
                Trace.TraceError($"Model directory {dir} does not exist");
                continue;
            }

            // find models in current dir that are of expected type
            _modelDirectories.Add(dir);
            var modelFiles = Directory.EnumerateFiles(dir)
                .Where(f =>
                {
                    var fi = new FileInfo(f);
                    return ExpectedModelFileTypes.Contains(fi.Extension);
                });

            _modelFiles.AddRange(modelFiles);
        }
    }

    public IEnumerable<string> ModelFiles => _modelFiles;
    public IEnumerable<string> ModelDirectories
    {
        get => _modelDirectories;
        set
        {
            _modelFiles.Clear();
            FindAndAddModels(value);
        }
    }

    public IEnumerable<LLamaWeights> GetLoadedModels()
    {
        return _modelCache.Values;
    }

    public bool TryGetLoadedModel(string modelId, out LLamaWeights model)
    {
        return _modelCache.TryGetValue(modelId, out model!);
    }

    public async Task<LLamaWeights> Load(string modelPath,
        Action<ModelParams>? modelConfigurator = null!,
        string modelId = "",
        CancellationToken cancellationToken = default)
    {
        var modelParams = new ModelParams(modelPath);
        modelConfigurator?.Invoke(modelParams);

        var model = await LLamaWeights.LoadFromFileAsync(modelParams, cancellationToken);
        if (string.IsNullOrWhiteSpace(modelId))
        {
            modelId = model.ModelName;
        }
        _modelCache.Add(modelId, model);
        return model;
    }

    public bool Unload(string modelId)
    {
        if (TryGetLoadedModel(modelId, out var model))
        {
            model.Dispose();
            return true;
        }

        return false;
    }

    public void UnloadAll()
    {
        foreach (var model in _modelCache.Values)
        {
            model.Dispose();
        }
        _modelCache.Clear();
    }
}
