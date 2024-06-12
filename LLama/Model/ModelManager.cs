using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LLama.Common;

namespace LLama.Model;

public enum ModelFileType
{
    GGUF
}

public class ModelFileMetadata
{
    public string FileName { get; init; } = string.Empty;
    public string FilePath { get; init; } = string.Empty;
    public ModelFileType ModelType { get; init; }
    public long SizeInBytes { get; init; } = 0;
}

internal interface IModelManager
{
    // Model Directories
    public IEnumerable<string> ModelDirectories { get; }
    public void AddDirectory(string directory);
    public bool RemoveDirectory(string directory);
    public void RemoveAllDirectories();

    // Model Files
    public IEnumerable<ModelFileMetadata> ModelFileList { get; }
    public IEnumerable<ModelFileMetadata> GetAvailableModelsFromDirectory(string directory);
    public bool TryGetModelFileMetadata(string fileName, out ModelFileMetadata modelMeta);

    // Model Load and Unload
    public Task<LLamaWeights> LoadModel(string modelPath,
        Action<ModelParams>? modelConfigurator = null!,
        string modelId = "",
        CancellationToken cancellationToken = default);
    public bool UnloadModel(string modelId);
    public void UnloadAllModels();

    public bool TryGetLoadedModel(string modeId, out LLamaWeights model);
    public IEnumerable<LLamaWeights> GetLoadedModels();
}

internal class ModelManager : IModelManager
{
    public static string[] ExpectedModelFileTypes = [
        ".gguf"
    ];

    // keys are directories, values are applicable models
    private readonly Dictionary<string, IEnumerable<ModelFileMetadata>> _availableModels = [];

    // model id/alias, to loaded model
    private readonly Dictionary<string, LLamaWeights> _loadedModelCache = [];

    public ModelManager(string[] directories)
    {
        FindAndAddModels(directories);
    }

    private void FindAndAddModels(params string[] dirs)
    {
        foreach (var dir in dirs)
        {
            var fullDirectoryPath = Path.GetFullPath(dir);

            if (!Directory.Exists(fullDirectoryPath))
            {
                Trace.TraceError($"Model directory '{fullDirectoryPath}' does not exist");
                continue;
            }

            if (_availableModels.ContainsKey(fullDirectoryPath))
            {
                Trace.TraceWarning($"Model directory '{fullDirectoryPath}' already probed");
                continue;
            }

            // find models in current dir that are of expected type
            List<ModelFileMetadata> directoryModelFiles = [];
            foreach (var file in Directory.EnumerateFiles(fullDirectoryPath))
            {
                if (!ExpectedModelFileTypes.Contains(Path.GetExtension(file)))
                {
                    continue;
                }

                // expected model file
                var fi = new FileInfo(file);
                directoryModelFiles.Add(new ModelFileMetadata
                {
                    FileName = fi.Name,
                    FilePath = fi.FullName,
                    ModelType = ModelFileType.GGUF,
                    SizeInBytes = fi.Length,
                });
            }

            _availableModels.Add(fullDirectoryPath, directoryModelFiles);
        }
    }

    public IEnumerable<ModelFileMetadata> ModelFileList
        => _availableModels.SelectMany(x => x.Value);
    public IEnumerable<string> ModelDirectories
        => _availableModels.Keys;

    public void AddDirectory(string directory)
    {
        FindAndAddModels(directory);
    }

    public bool RemoveDirectory(string directory)
    {
        return _availableModels.Remove(directory);
    }

    public void RemoveAllDirectories()
    {
        _availableModels.Clear();
    }

    public IEnumerable<ModelFileMetadata> GetAvailableModelsFromDirectory(string directory)
    {
        var dirPath = Path.GetFullPath(directory);
        return _availableModels.TryGetValue(dirPath, out var dirModels)
            ? dirModels
            : [];
    }

    public bool TryGetModelFileMetadata(string fileName, out ModelFileMetadata modelMeta)
    {
        var filePath = Path.GetFullPath(fileName);
        modelMeta = ModelFileList.FirstOrDefault(f => f.FilePath == filePath)!;
        return modelMeta != null;
    }

    public IEnumerable<LLamaWeights> GetLoadedModels()
    {
        return _loadedModelCache.Values;
    }

    public bool TryGetLoadedModel(string modelId, out LLamaWeights model)
    {
        return _loadedModelCache.TryGetValue(modelId, out model!);
    }

    public async Task<LLamaWeights> LoadModel(string modelPath,
        Action<ModelParams>? modelConfigurator = null!,
        string modelId = "",
        CancellationToken cancellationToken = default)
    {
        // Configure model params
        var modelParams = new ModelParams(modelPath);
        modelConfigurator?.Invoke(modelParams);

        // load and cache
        var model = await LLamaWeights.LoadFromFileAsync(modelParams, cancellationToken);
        if (string.IsNullOrWhiteSpace(modelId))
        {
            modelId = model.ModelName;
        }
        _loadedModelCache.Add(modelId, model);
        return model;
    }
    
    public bool UnloadModel(string modelId)
    {
        if (TryGetLoadedModel(modelId, out var model))
        {
            model.Dispose();
            return _loadedModelCache.Remove(modelId);
        }
        return false;
    }

    public void UnloadAllModels()
    {
        foreach (var model in _loadedModelCache.Values)
        {
            model.Dispose();
        }
        _loadedModelCache.Clear();
    }
}
