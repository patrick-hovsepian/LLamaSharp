using LLama.Common;
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

        var expected = dirs.First()!.Contains(Constants.ModelDirectory);
        Assert.True(expected);
    }

    [Fact]
    public void ModelDirectories_AddDirectory_DoesntDuplicate()
    {
        for (var i = 0; i < 10; i++)
        {
            TestableModelManager.AddDirectory(Constants.ModelDirectory);

            var dirs = TestableModelManager.ModelDirectories;
            Assert.Single(dirs);
            var expected = dirs.First()!.Contains(Constants.ModelDirectory);
            Assert.True(expected);
        }
    }

    [Fact]
    public void ModelFiles_IsCorrect()
    {
        var files = TestableModelManager.ModelFileList;
        Assert.Equal(4, files.Count());
    }

    [Fact]
    public async Task LoadModel_LoadsAndCaches()
    {
        var modelToLoad = TestableModelManager.ModelFileList
            .First(f => f.FileName.Contains("llama-2-7b"));

        var model = await TestableModelManager.LoadModel(modelToLoad.FilePath, (m) =>
        {
            m.GpuLayerCount = 1;
            m.Seed = 12;
        });

        Assert.Single(TestableModelManager.GetLoadedModels());

        var isLoaded = TestableModelManager.TryGetLoadedModel(model.ModelName, out var cachedModel);
        Assert.True(isLoaded);

        // unload
        Assert.True(TestableModelManager.UnloadModel(model.ModelName));

        Assert.Throws<ObjectDisposedException>(() => {
            _ = model.CreateContext(new ModelParams(modelToLoad.FilePath));
        });
    }
}
