use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

use quark::{Command, CommandError, Quark, Result as QuarkResult};

/// Actions emitted by Quark commands for the main renderer thread.
pub enum HelioAction {
    SetDebugMode(u32),
    SetEditorMode(bool),
    SetDebugDepthTest(bool),
    DebugClear,
}

/// Bridge that holds a Quark command registry for a Helio renderer.
pub struct HelioCommandBridge {
    pub registry: Arc<Mutex<Quark>>,
}

impl HelioCommandBridge {
    /// Create a new command bridge and action receiver.
    pub fn new() -> (Self, Receiver<HelioAction>) {
        let (tx, rx) = mpsc::channel();
        let mut registry = Quark::new();
        register_helio_commands(&mut registry, tx.clone());

        (
            Self {
                registry: Arc::new(Mutex::new(registry)),
            },
            rx,
        )
    }

    /// Execute a command string synchronously.
    pub fn run(&self, input: &str) -> QuarkResult<()> {
        let guard = self
            .registry
            .lock()
            .map_err(|e| CommandError::ExecutionError(format!("Mutex poisoned: {}", e)))?;
        guard.run(input)
    }

    /// Execute a command string asynchronously.
    pub async fn run_async(&self, input: &str) -> QuarkResult<()> {
        let guard = self
            .registry
            .lock()
            .map_err(|e| CommandError::ExecutionError(format!("Mutex poisoned: {}", e)))?;
        guard.run_async(input).await
    }
}

/// Register built-in Helio commands that emit renderer actions.
pub fn register_helio_commands(registry: &mut Quark, sender: Sender<HelioAction>) {
    registry.register_command(SetDebugModeCommand {
        sender: sender.clone(),
    });
    registry.register_command(SetEditorModeCommand {
        sender: sender.clone(),
    });
    registry.register_command(SetDebugDepthTestCommand {
        sender: sender.clone(),
    });
    registry.register_command(DebugClearCommand { sender });
    registry.register_command(HelpCommand {});
}

struct SetDebugModeCommand {
    sender: Sender<HelioAction>,
}

impl Command for SetDebugModeCommand {
    fn name(&self) -> &str {
        "set_debug_mode"
    }

    fn syntax(&self) -> &str {
        "set_debug_mode <mode>"
    }

    fn short(&self) -> &str {
        "Set Helio renderer debug visualization mode"
    }

    fn docs(&self) -> &str {
        "Usage: set_debug_mode 0|10|11 (0=normal, 10=shadow heatmap, 11=light-space depth)"
    }

    fn execute(&self, args: Vec<String>) -> QuarkResult<()> {
        if args.len() != 1 {
            return Err(CommandError::ArgumentCountMismatch {
                expected: 1,
                got: args.len(),
            });
        }

        let mode = args[0]
            .parse::<u32>()
            .map_err(|_| CommandError::TypeConversionError {
                arg: args[0].clone(),
                target_type: "u32",
            })?;

        self.sender
            .send(HelioAction::SetDebugMode(mode))
            .map_err(|e| CommandError::ExecutionError(format!("Channel send failed: {}", e)))?;

        Ok(())
    }
}

struct SetEditorModeCommand {
    sender: Sender<HelioAction>,
}

impl Command for SetEditorModeCommand {
    fn name(&self) -> &str {
        "set_editor_mode"
    }

    fn syntax(&self) -> &str {
        "set_editor_mode <true|false>"
    }

    fn short(&self) -> &str {
        "Enable/disable the Helio editor mode"
    }

    fn docs(&self) -> &str {
        "Usage: set_editor_mode true | false"
    }

    fn execute(&self, args: Vec<String>) -> QuarkResult<()> {
        if args.len() != 1 {
            return Err(CommandError::ArgumentCountMismatch {
                expected: 1,
                got: args.len(),
            });
        }

        let enabled = args[0]
            .parse::<bool>()
            .map_err(|_| CommandError::TypeConversionError {
                arg: args[0].clone(),
                target_type: "bool",
            })?;

        self.sender
            .send(HelioAction::SetEditorMode(enabled))
            .map_err(|e| CommandError::ExecutionError(format!("Channel send failed: {}", e)))?;

        Ok(())
    }
}

struct SetDebugDepthTestCommand {
    sender: Sender<HelioAction>,
}

impl Command for SetDebugDepthTestCommand {
    fn name(&self) -> &str {
        "set_debug_depth_test"
    }

    fn syntax(&self) -> &str {
        "set_debug_depth_test <true|false>"
    }

    fn short(&self) -> &str {
        "Enable/disable debug depth testing"
    }

    fn docs(&self) -> &str {
        "Usage: set_debug_depth_test true | false"
    }

    fn execute(&self, args: Vec<String>) -> QuarkResult<()> {
        if args.len() != 1 {
            return Err(CommandError::ArgumentCountMismatch {
                expected: 1,
                got: args.len(),
            });
        }

        let enabled = args[0]
            .parse::<bool>()
            .map_err(|_| CommandError::TypeConversionError {
                arg: args[0].clone(),
                target_type: "bool",
            })?;

        self.sender
            .send(HelioAction::SetDebugDepthTest(enabled))
            .map_err(|e| CommandError::ExecutionError(format!("Channel send failed: {}", e)))?;

        Ok(())
    }
}

struct DebugClearCommand {
    sender: Sender<HelioAction>,
}

impl Command for DebugClearCommand {
    fn name(&self) -> &str {
        "debug_clear"
    }

    fn syntax(&self) -> &str {
        "debug_clear"
    }

    fn short(&self) -> &str {
        "Clear all Helio per-frame debug drawing"
    }

    fn docs(&self) -> &str {
        "Usage: debug_clear"
    }

    fn execute(&self, args: Vec<String>) -> QuarkResult<()> {
        if !args.is_empty() {
            return Err(CommandError::ArgumentCountMismatch {
                expected: 0,
                got: args.len(),
            });
        }

        self.sender
            .send(HelioAction::DebugClear)
            .map_err(|e| CommandError::ExecutionError(format!("Channel send failed: {}", e)))?;

        Ok(())
    }
}

struct HelpCommand;

impl Command for HelpCommand {
    fn name(&self) -> &str {
        "help"
    }

    fn syntax(&self) -> &str {
        "help"
    }

    fn short(&self) -> &str {
        "Display available commands"
    }

    fn docs(&self) -> &str {
        "Usage: help\nPrints command list and usage."
    }

    fn execute(&self, args: Vec<String>) -> QuarkResult<()> {
        if !args.is_empty() {
            return Err(CommandError::ArgumentCountMismatch {
                expected: 0,
                got: args.len(),
            });
        }

        println!("Available commands:");
        println!("  set_debug_mode <0|10|11> - Set renderer debug mode");
        println!("  set_editor_mode <true|false> - Enable/disable editor helpers");
        println!("  set_debug_depth_test <true|false> - Toggle debug depth test");
        println!("  debug_clear - Clear debug lines");
        println!("  help - Show this list");

        Ok(())
    }
}
