using LLama.Model;

namespace LLama.Unittest;

public class ModelManagerTests
{
    private readonly ModelManager TestableModelManager;

    public ModelManagerTests()
    {
        TestableModelManager = new([Constants.ModelDirectory]);
    }

    [Fact]
    public void ModelDirectories_IsCorrect()
    {
        var dirs = TestableModelManager.ModelDirectories;
        Assert.Single(dirs);

        Assert.Equal(Constants.ModelDirectory, dirs.First());
    }

    [Fact]
    public void ModelFiles_IsCorrect()
    {
        var files = TestableModelManager.ModelFiles;
        Assert.Equal(4, files.Count());
    }

    [Fact]
    public async Task LoadModel_LoadsAndCaches()
    {
        var modelToLoad = TestableModelManager.ModelFiles
            .First(f => f.Contains("llama-2-7b"));

        var model = await TestableModelManager.Load(modelToLoad, (m) =>
        {
            m.GpuLayerCount = 1;
            m.Seed = 12;
        });

        Assert.Single(TestableModelManager.GetLoadedModels());

        var isLoaded = TestableModelManager.TryGetLoadedModel(model.ModelName, out var cachedModel);
        Assert.True(isLoaded);

        // unload
        Assert.True(TestableModelManager.Unload(model.ModelName));
    }
}
