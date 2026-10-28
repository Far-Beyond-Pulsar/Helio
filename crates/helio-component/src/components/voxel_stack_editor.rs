//! Property editor for [`VoxelTerrainStack`]: the whole terrain layer stack
//! as one value, so a preset replaces all of it and undo is one step.
//!
//! The stack's layers and material rules are ordered lists (move up and
//! down, remove, add); every field of a layer, a rule, the caves, the
//! cliffs and the materials is edited by the editor registered for its
//! type, through reflection (the shared pieces in `reflected_list`).

use std::sync::{Arc, Mutex};

use gpui::{
    AnyElement, App, Entity, SharedString, Subscription, WeakEntity, Window, prelude::*, px,
};
use pulsar_reflection::{BoundPropertyEditor, EngineClass, PropertyEditorArgs, PropertyWriteBack};
use ui::button::{Button, ButtonVariants as _};
use ui::dropdown::{Dropdown, DropdownEvent, DropdownState};
use ui::{ActiveTheme as _, IconName, Sizable as _, v_flex};

use super::reflected_list::{FieldRows, ListEdit, apply_list_edit, item_card};
use super::{VoxelLayerKind, VoxelMaterialRule, VoxelTerrainLayer, VoxelTerrainStack};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum List {
    Layers,
    Rules,
}

pub struct VoxelTerrainStackEditor {
    label: String,
    id: String,
    current: Arc<Mutex<VoxelTerrainStack>>,
    write_back: PropertyWriteBack,
    presets: Entity<DropdownState<Vec<SharedString>>>,
    materials: FieldRows,
    caves: FieldRows,
    cliffs: FieldRows,
    layers: Vec<FieldRows>,
    rules: Vec<FieldRows>,
    this: WeakEntity<Self>,
    _subs: Vec<Subscription>,
}

impl VoxelTerrainStackEditor {
    fn new(
        args: &PropertyEditorArgs<'_>,
        window: &mut Window,
        cx: &mut gpui::Context<Self>,
    ) -> Self {
        let value = args
            .current_value
            .downcast_ref::<VoxelTerrainStack>()
            .cloned()
            .unwrap_or_default();
        let id = format!(
            "stack-{}-{}-{}",
            args.id_prefix, args.class_name, args.prop_name
        );
        let current = Arc::new(Mutex::new(value.clone()));
        let write_back = args.write_back.clone();
        let names: Vec<SharedString> = VoxelTerrainStack::PRESETS
            .iter()
            .map(|n| SharedString::from(*n))
            .collect();
        let presets = cx.new(|cx| DropdownState::new(names, None, window, cx));
        let subs = vec![cx.subscribe_in(
            &presets,
            window,
            |this: &mut Self, _, event: &DropdownEvent<Vec<SharedString>>, window, cx| {
                let DropdownEvent::Confirm(Some(name)) = event else {
                    return;
                };
                if let Some(stack) = VoxelTerrainStack::preset(name) {
                    this.write(stack, window, cx);
                }
                this.presets
                    .update(cx, |state, cx| state.set_selected_index(None, window, cx));
            },
        )];
        let this = cx.weak_entity();
        let materials = Self::rows(
            &id,
            "materials",
            &this,
            &current,
            &write_back,
            |s| Some(s.clone()),
            |s, v| replace(s, v),
            &value,
            window,
            cx,
        );
        let caves = Self::rows(
            &id,
            "caves",
            &this,
            &current,
            &write_back,
            |s| Some(s.volumes.caves.clone()),
            |s, v| replace(&mut s.volumes.caves, v),
            &value.volumes.caves,
            window,
            cx,
        );
        let cliffs = Self::rows(
            &id,
            "cliffs",
            &this,
            &current,
            &write_back,
            |s| Some(s.volumes.cliffs.clone()),
            |s, v| replace(&mut s.volumes.cliffs, v),
            &value.volumes.cliffs,
            window,
            cx,
        );
        let mut editor = Self {
            label: args.display_name.to_string(),
            id,
            current,
            write_back,
            presets,
            materials,
            caves,
            cliffs,
            layers: Vec::new(),
            rules: Vec::new(),
            this,
            _subs: subs,
        };
        editor.sync_lists(window, cx);
        editor
    }

    /// Field rows of the part of the stack `get` reads and `set` writes.
    /// `set` returns true when it changed more than the edited field (a
    /// layer's new kind brings that kind's defaults): every row then shows
    /// the new values.
    #[allow(clippy::too_many_arguments)]
    fn rows<T: EngineClass + Clone>(
        id: &str,
        part: &str,
        this: &WeakEntity<Self>,
        current: &Arc<Mutex<VoxelTerrainStack>>,
        write_back: &PropertyWriteBack,
        get: impl Fn(&VoxelTerrainStack) -> Option<T> + Send + Sync + 'static,
        set: impl Fn(&mut VoxelTerrainStack, T) -> bool + Send + Sync + 'static,
        value: &T,
        window: &mut Window,
        cx: &mut App,
    ) -> FieldRows {
        let (read, write) = (current.clone(), current.clone());
        let write_back = write_back.clone();
        let this = this.clone();
        FieldRows::new(
            value,
            &format!("{id}-{part}"),
            "VoxelTerrainStack",
            |_, _| true,
            move || read.lock().ok().and_then(|stack| get(&stack)),
            move |v, window, cx| {
                let (stack, reshaped) = {
                    let Ok(mut stack) = write.lock() else { return };
                    let reshaped = set(&mut stack, v);
                    (stack.clone(), reshaped)
                };
                write_back(Box::new(stack), window, cx);
                if reshaped {
                    // After the editor that wrote returns: it is mid-event.
                    let this = this.clone();
                    window.defer(cx, move |window, cx| {
                        let _ = this.update(cx, |this, cx| {
                            this.refresh(window, cx);
                            cx.notify();
                        });
                    });
                }
            },
            window,
            cx,
        )
    }

    /// One row set per layer and rule (rebuilt when a list changes length;
    /// rows follow indices, so moves only refresh values).
    fn sync_lists(&mut self, window: &mut Window, cx: &mut App) {
        let value = self.current.lock().map(|s| s.clone()).unwrap_or_default();
        if self.layers.len() != value.layers.len() {
            self.layers = (0..value.layers.len())
                .map(|i| {
                    Self::rows(
                        &self.id,
                        &format!("layer{i}"),
                        &self.this,
                        &self.current,
                        &self.write_back,
                        move |s| s.layers.get(i).cloned(),
                        move |s, v| s.layers.get_mut(i).is_some_and(|layer| set_layer(layer, v)),
                        &value.layers[i],
                        window,
                        cx,
                    )
                })
                .collect();
        }
        if self.rules.len() != value.rules.len() {
            self.rules = (0..value.rules.len())
                .map(|i| {
                    Self::rows(
                        &self.id,
                        &format!("rule{i}"),
                        &self.this,
                        &self.current,
                        &self.write_back,
                        move |s| s.rules.get(i).cloned(),
                        move |s, v| s.rules.get_mut(i).is_some_and(|rule| replace(rule, v)),
                        &value.rules[i],
                        window,
                        cx,
                    )
                })
                .collect();
        }
    }

    fn refresh(&mut self, window: &mut Window, cx: &mut App) {
        self.sync_lists(window, cx);
        for rows in [&self.materials, &self.caves, &self.cliffs]
            .into_iter()
            .chain(&self.layers)
            .chain(&self.rules)
        {
            rows.refresh(window, cx);
        }
    }

    /// Replace the whole stack (a preset, a list edit) and write it back.
    fn write(
        &mut self,
        stack: VoxelTerrainStack,
        window: &mut Window,
        cx: &mut gpui::Context<Self>,
    ) {
        if let Ok(mut current) = self.current.lock() {
            *current = stack.clone();
        }
        self.refresh(window, cx);
        (self.write_back)(Box::new(stack), window, cx);
        cx.notify();
    }

    fn edit_list(
        &mut self,
        list: List,
        edit: ListEdit,
        window: &mut Window,
        cx: &mut gpui::Context<Self>,
    ) {
        let mut stack = self.current.lock().map(|s| s.clone()).unwrap_or_default();
        match list {
            List::Layers => apply_list_edit(&mut stack.layers, edit, || {
                VoxelTerrainLayer::new(VoxelLayerKind::Hills)
            }),
            List::Rules => apply_list_edit(&mut stack.rules, edit, VoxelMaterialRule::default),
        }
        self.write(stack, window, cx);
    }

    /// Accept a value that changed elsewhere (undo, a blueprint, a preset).
    fn set_value(
        &mut self,
        value: &VoxelTerrainStack,
        window: &mut Window,
        cx: &mut gpui::Context<Self>,
    ) {
        let changed = self
            .current
            .lock()
            .map(|current| *current != *value)
            .unwrap_or(true);
        if !changed {
            return;
        }
        if let Ok(mut current) = self.current.lock() {
            *current = value.clone();
        }
        self.refresh(window, cx);
        cx.notify();
    }

    fn heading(title: &str, cx: &App) -> AnyElement {
        gpui::div()
            .pt_2()
            .text_sm()
            .text_color(cx.theme().foreground)
            .child(title.to_string())
            .into_any_element()
    }

    fn list_items(
        &self,
        list: List,
        titles: Vec<String>,
        cx: &mut gpui::Context<Self>,
    ) -> Vec<AnyElement> {
        let rows = match list {
            List::Layers => &self.layers,
            List::Rules => &self.rules,
        };
        let tag = if list == List::Layers {
            "layer"
        } else {
            "rule"
        };
        rows.iter()
            .zip(titles)
            .enumerate()
            .map(|(i, (rows, title))| {
                item_card(
                    &format!("{}-{tag}", self.id),
                    i,
                    title,
                    rows.views(),
                    move |this: &mut Self, edit, window, cx| this.edit_list(list, edit, window, cx),
                    cx,
                )
            })
            .collect()
    }

    fn add_button(&self, list: List, label: &str, cx: &mut gpui::Context<Self>) -> AnyElement {
        let tag = if list == List::Layers {
            "layer"
        } else {
            "rule"
        };
        Button::new(SharedString::from(format!("{}-add-{tag}", self.id)))
            .icon(IconName::Plus)
            .label(label.to_string())
            .xsmall()
            .ghost()
            .on_click(cx.listener(move |this, _, window, cx| {
                this.edit_list(list, ListEdit::Add, window, cx)
            }))
            .into_any_element()
    }
}

/// Why the generator would reject the stack (its own validation), if it
/// would: the planet does not build until it is fixed.
pub fn validation_error(stack: &VoxelTerrainStack) -> Option<String> {
    let json = serde_json::to_value(stack).ok()?;
    match serde_json::from_value::<helio_pass_voxel_planet::layers::TerrainLayers>(json) {
        Ok(layers) => layers.validate().err(),
        Err(error) => Some(error.to_string()),
    }
}

/// Store an edited value; nothing else changes.
fn replace<T>(slot: &mut T, value: T) -> bool {
    *slot = value;
    false
}

/// Store an edited layer. A new kind starts from that kind's defaults (the
/// old numbers mean something else, or nothing, for it), keeping whether
/// the layer is on and where it applies; returns true then.
pub fn set_layer(layer: &mut VoxelTerrainLayer, edited: VoxelTerrainLayer) -> bool {
    if edited.kind == layer.kind {
        *layer = edited;
        return false;
    }
    *layer = VoxelTerrainLayer {
        enabled: edited.enabled,
        mask: edited.mask,
        ..VoxelTerrainLayer::new(edited.kind)
    };
    true
}

/// A layer's or rule's title: its position and kind or material.
fn layer_title(index: usize, layer: &VoxelTerrainLayer) -> String {
    format!(
        "{}. {:?}{}",
        index + 1,
        layer.kind,
        if layer.enabled { "" } else { " (off)" }
    )
}

fn rule_title(index: usize, rule: &VoxelMaterialRule) -> String {
    format!("{}. {:?}", index + 1, rule.material)
}

impl gpui::Render for VoxelTerrainStackEditor {
    fn render(
        &mut self,
        _window: &mut Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        let value = self.current.lock().map(|s| s.clone()).unwrap_or_default();
        let layer_titles = value
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| layer_title(i, l))
            .collect();
        let rule_titles = value
            .rules
            .iter()
            .enumerate()
            .map(|(i, r)| rule_title(i, r))
            .collect();
        let layers = self.list_items(List::Layers, layer_titles, cx);
        let rules = self.list_items(List::Rules, rule_titles, cx);
        let error = validation_error(&value).map(|error| {
            gpui::div()
                .text_sm()
                .text_color(cx.theme().danger)
                .child(format!("Not generated: {error}"))
        });
        v_flex()
            .w_full()
            .gap_1()
            .child(pulsar_reflection::prims::editor_row(
                &self.label,
                Dropdown::new(&self.presets)
                    .xsmall()
                    .w(px(140.0))
                    .placeholder("Apply preset"),
                cx,
            ))
            .children(error)
            .child(Self::heading("Layers (in order)", cx))
            .children(layers)
            .child(self.add_button(List::Layers, "Add layer", cx))
            .child(Self::heading("Materials", cx))
            .children(self.materials.views())
            .child(Self::heading("Material rules (Rules style, in order)", cx))
            .children(rules)
            .child(self.add_button(List::Rules, "Add rule", cx))
            .child(Self::heading("Caves", cx))
            .children(self.caves.views())
            .child(Self::heading("Cliffs", cx))
            .children(self.cliffs.views())
    }
}

fn voxel_terrain_stack_editor(
    args: &PropertyEditorArgs<'_>,
    window: &mut Window,
    cx: &mut App,
) -> BoundPropertyEditor {
    let entity = cx.new(|cx| VoxelTerrainStackEditor::new(args, window, cx));
    BoundPropertyEditor::new(
        entity,
        |editor: &mut VoxelTerrainStackEditor, value: &VoxelTerrainStack, window, cx| {
            editor.set_value(value, window, cx)
        },
    )
}

pulsar_reflection::inventory::submit! {
    pulsar_reflection::UiPropertyEditorHint {
        type_id: std::any::TypeId::of::<VoxelTerrainStack>(),
        fn_ptr: pulsar_reflection::erase_property_editor_fn_ptr(voxel_terrain_stack_editor),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_stacks_explain_themselves() {
        assert_eq!(validation_error(&VoxelTerrainStack::earth()), None);
        let mut stack = VoxelTerrainStack::earth();
        stack.layers.swap(0, 1);
        assert!(
            validation_error(&stack).is_some_and(|e| e.contains("Warp")),
            "{:?}",
            validation_error(&stack)
        );
    }
}
