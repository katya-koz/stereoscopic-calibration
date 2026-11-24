#include "App.h"
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include "Utils.h"

int main(int argc, char** argv)
{
    std::filesystem::path configPath = Utils::getExecutableDirectory() / "config.json";
    bool debugNoTracker = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--config")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Error: --config requires a file path\n";
                return -1;
            }

            configPath = argv[++i];
        }
        else if (arg == "--debug")
        {
            debugNoTracker = true;
        }
        else
        {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            std::cerr << "Usage: App.exe [--config <config.json>] [--debug]\n";
            return -1;
        }
    }

    try
    {
        App app;
        app.setDebugNoTracker(debugNoTracker);

        if (!app.init(configPath.string()))
        {
            std::cerr << "Failed to initialize app with config: "
                << configPath << '\n';
            return -1;
        }

        app.run();
    }
    catch (const std::exception& e) // the renderer reports Vulkan / shader / texture problems by throwing
    {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return -1;
    }

    return 0;
}