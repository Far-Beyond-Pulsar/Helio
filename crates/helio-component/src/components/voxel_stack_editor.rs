//! Property editor for [`VoxelTerrainStack`]: the whole terrain layer stack
//! as one value, so a preset replaces all of it and undo is one step.
//!
//! The stack's layers and material rules are ordered lists (move up and
//! down, remove, add); every field of a layer, a rule, the caves, the
//! overhangs and the materials is edited by the editor registered for its
//! type, through reflection ([`FieldRows`]).

use std::any::Any;
use std::sync::{Arc, Mutex};

use gpui::{prelude::*, px, AnyElement, App, Entity, SharedString, Subscription, Window};
use pulsar_reflection::{BoundPropertyEditor, EngineClass, PropertyEditorArgs, PropertyEditorFactory, PropertyMetadata, PropertyWriteBack};
use ui::button::{Button, ButtonVariants as _};
use ui::dropdown::{Dropdown, DropdownEvent, DropdownState};
use ui::{h_flex, v_flex, ActiveTheme as _, IconName, Sizable as _};

use super::{VoxelMaterialRule, VoxelTerrainLayer, VoxelTerrainStack};

/// The editor registered for a property type, if any.
fn editor_factory(type_id: std::any::TypeId) -> Option<PropertyEditorFactory> {
    pulsar_reflection::inventory::iter::<pulsar_reflection::UiPropertyEditorHint>
        .into_iter()
        .find(|hint| hint.type_id == type_id)
        // SAFETY: `erase_property_editor_fn_ptr` only accepts a
        // `PropertyEditorFactory`, so every submitted pointer has that type
        // (the same contract the inspector's registry relies on).
        .map(|hint| unsafe { std::mem::transmute::<fn(), PropertyEditorFactory>(hint.fn_ptr) })
}

/// Editors for the reflected properties of one value (a layer, a rule, the
/// caves...), each writing the whole value back through `put`. Properties
/// without a registered editor (lists, nested structs) are skipped.
struct FieldRows {
    props: Arc<Vec<PropertyMetadata>>,
    editors: Vec<(usize, BoundPropertyEditor)>,
    /// Pushes the current value into every editor.
    refresh: Arc<dyn Fn(&[(usize, BoundPropertyEditor)], &mut Window, &mut App) + Send + Sync>,
}

impl FieldRows {
    fn new<T: EngineClass + Clone>(
        value: &T,
        id: &str,
        get: impl Fn() -> Option<T> + Send + Sync + 'static,
        put: impl Fn(T, &mut Window, &mut App) + Send + Sync + 'static,
        window: &mut Window,
        cx: &mut App,
    ) -> Self {
        let props = Arc::new(value.get_properties());
        let get = Arc::new(get);
        let put = Arc::new(put);
        let mut editors = Vec::new();
        for (index, prop) in props.iter().enumerate() {
            let Some(factory) = editor_factory(prop.type_info.type_id) else { continue };
            let current = (prop.getter)(value);
            let write_back: PropertyWriteBack = {
                let (props, get, put) = (props.clone(), get.clone(), put.clone());
                Arc::new(move |new: Box<dyn Any + Send>, window: &mut Window, cx: &mut App| {
                    let Some(mut value) = get() else { return };
                    (props[index].setter)(&mut value, new);
                    put(value, window, cx);
                })
            };
            let args = PropertyEditorArgs {
                id_prefix: id,
                class_name: "VoxelTerrainStack",
                display_name: &prop.display_name,
                prop_name: prop.name,
                type_info: prop.type_info,
                current_value: current.as_ref(),
                write_back,
            };
            editors.push((index, factory(&args, window, cx)));
        }
        let refresh: Arc<dyn Fn(&[(usize, BoundPropertyEditor)], &mut Window, &mut App) + Send + Sync> = {
            let props = props.clone();
            Arc::new(move |editors: &[(usize, BoundPropertyEditor)], window: &mut Window, cx: &mut App| {
                let Some(value) = get() else { return };
                for (index, editor) in editors {
                    (editor.set_value)((props[*index].getter)(&value).as_ref(), window, cx);
                }
            })
        };
        Self { props, editors, refresh }
    }

    fn refresh(&self, window: &mut Window, cx: &mut App) {
        (self.refresh)(&self.editors, window, cx);
    }

    fn views(&self) -> impl Iterator<Item = AnyElement> + '_ {
        let _ = &self.props;
        self.editors.iter().map(|(_, editor)| editor.view.clone().into_any_element())
    }
}

/// Moves, removals and additions of a list item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListEdit {
    Up(usize),
    Down(usize),
    Remove(usize),
    Add,
}

/// Apply `edit` to `items` (out-of-range moves do nothing); `new` makes an
/// added item.
pub fn apply_list_edit<T>(items: &mut Vec<T>, edit: ListEdit, new: impl FnOnce() -> T) {
    match edit {
        ListEdit::Up(i) if i > 0 && i < items.len() => items.swap(i - 1, i),
        ListEdit::Down(i) if i + 1 < items.len() => items.swap(i, i + 1),
        ListEdit::Remove(i) if i < items.len() => {
            items.remove(i);
        }
        ListEdit::Add => items.push(new()),
        _ => {}
    }
}

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
    overhangs: FieldRows,
    layers: Vec<FieldRows>,
    rules: Vec<FieldRows>,
    _subs: Vec<Subscription>,
}

impl VoxelTerrainStackEditor {
    fn new(args: &PropertyEditorArgs<'_>, window: &mut Window, cx: &mut gpui::Context<Self>) -> Self {
        let value = args.current_value.downcast_ref::<VoxelTerrainStack>().cloned().unwrap_or_default();
        let id = format!("stack-{}-{}-{}", args.id_prefix, args.class_name, args.prop_name);
        let current = Arc::new(Mutex::new(value.clone()));
        let write_back = args.write_back.clone();
        let names: Vec<SharedString> = VoxelTerrainStack::PRESETS.iter().map(|n| SharedString::from(*n)).collect();
        let presets = cx.new(|cx| DropdownState::new(names, None, window, cx));
        let subs = vec![cx.subscribe_in(&presets, window, |this: &mut Self, _, event: &DropdownEvent<Vec<SharedString>>, window, cx| {
            let DropdownEvent::Confirm(Some(name)) = event else { return };
            if let Some(stack) = VoxelTerrainStack::preset(name) {
                this.write(stack, window, cx);
            }
            this.presets.update(cx, |state, cx| state.set_selected_index(None, window, cx));
        })];
        let materials = Self::rows(&id, "materials", &current, &write_back, |s| Some(s.clone()), |s, v| *s = v, &value, window, cx);
        let caves = Self::rows(&id, "caves", &current, &write_back, |s| Some(s.caves.clone()), |s, v| s.caves = v, &value.caves, window, cx);
        let overhangs =
            Self::rows(&id, "overhangs", &current, &write_back, |s| Some(s.overhangs.clone()), |s, v| s.overhangs = v, &value.overhangs, window, cx);
        let mut editor = Self {
            label: args.display_name.to_string(),
            id,
            current,
            write_back,
            presets,
            materials,
            caves,
            overhangs,
            layers: Vec::new(),
            rules: Vec::new(),
            _subs: subs,
        };
        editor.sync_lists(window, cx);
        editor
    }

    /// Field rows of the part of the stack `get` reads and `set` writes.
    #[allow(clippy::too_many_arguments)]
    fn rows<T: EngineClass + Clone>(
        id: &str,
        part: &str,
        current: &Arc<Mutex<VoxelTerrainStack>>,
        write_back: &PropertyWriteBack,
        get: impl Fn(&VoxelTerrainStack) -> Option<T> + Send + Sync + 'static,
        set: impl Fn(&mut VoxelTerrainStack, T) + Send + Sync + 'static,
        value: &T,
        window: &mut Window,
        cx: &mut App,
    ) -> FieldRows {
        let (read, write) = (current.clone(), current.clone());
        let write_back = write_back.clone();
        FieldRows::new(
            value,
            &format!("{id}-{part}"),
            move || read.lock().ok().and_then(|stack| get(&stack)),
            move |v, window, cx| {
                let stack = {
                    let Ok(mut stack) = write.lock() else { return };
                    set(&mut stack, v);
                    stack.clone()
                };
                write_back(Box::new(stack), window, cx);
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
                        &self.current,
                        &self.write_back,
                        move |s| s.layers.get(i).cloned(),
                        move |s, v| {
                            if let Some(layer) = s.layers.get_mut(i) {
                                *layer = v;
                            }
                        },
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
                        &self.current,
                        &self.write_back,
                        move |s| s.rules.get(i).cloned(),
                        move |s, v| {
                            if let Some(rule) = s.rules.get_mut(i) {
                                *rule = v;
                            }
                        },
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
        for rows in [&self.materials, &self.caves, &self.overhangs].into_iter().chain(&self.layers).chain(&self.rules) {
            rows.refresh(window, cx);
        }
    }

    /// Replace the whole stack (a preset, a list edit) and write it back.
    fn write(&mut self, stack: VoxelTerrainStack, window: &mut Window, cx: &mut gpui::Context<Self>) {
        if let Ok(mut current) = self.current.lock() {
            *current = stack.clone();
        }
        self.refresh(window, cx);
        (self.write_back)(Box::new(stack), window, cx);
        cx.notify();
    }

    fn edit_list(&mut self, list: List, edit: ListEdit, window: &mut Window, cx: &mut gpui::Context<Self>) {
        let mut stack = self.current.lock().map(|s| s.clone()).unwrap_or_default();
        match list {
            List::Layers => apply_list_edit(&mut stack.layers, edit, VoxelTerrainLayer::default),
            List::Rules => apply_list_edit(&mut stack.rules, edit, VoxelMaterialRule::default),
        }
        self.write(stack, window, cx);
    }

    /// Accept a value that changed elsewhere (undo, a blueprint, a preset).
    fn set_value(&mut self, value: &VoxelTerrainStack, window: &mut Window, cx: &mut gpui::Context<Self>) {
        let changed = self.current.lock().map(|current| *current != *value).unwrap_or(true);
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
        gpui::div().pt_2().text_sm().text_color(cx.theme().foreground).child(title.to_string()).into_any_element()
    }

    fn list_items(&self, list: List, titles: Vec<String>, cx: &mut gpui::Context<Self>) -> Vec<AnyElement> {
        let rows = match list {
            List::Layers => &self.layers,
            List::Rules => &self.rules,
        };
        let tag = if list == List::Layers { "layer" } else { "rule" };
        rows.iter()
            .zip(titles)
            .enumerate()
            .map(|(i, (rows, title))| {
                let button = |name: &str, icon: IconName, tooltip: &str, edit: ListEdit, cx: &mut gpui::Context<Self>| {
                    Button::new(SharedString::from(format!("{}-{tag}{i}-{name}", self.id)))
                        .icon(icon)
                        .xsmall()
                        .ghost()
                        .tooltip(tooltip.to_string())
                        .on_click(cx.listener(move |this, _, window, cx| this.edit_list(list, edit, window, cx)))
                };
                v_flex()
                    .w_full()
                    .gap_1()
                    .p_1()
                    .rounded_md()
                    .border_1()
                    .border_color(cx.theme().border)
                    .child(
                        h_flex()
                            .w_full()
                            .justify_between()
                            .child(gpui::div().text_sm().child(title))
                            .child(
                                h_flex()
                                    .gap_1()
                                    .child(button("up", IconName::ArrowUp, "Move up", ListEdit::Up(i), cx))
                                    .child(button("down", IconName::ArrowDown, "Move down", ListEdit::Down(i), cx))
                                    .child(button("remove", IconName::Trash, "Remove", ListEdit::Remove(i), cx)),
                            ),
                    )
                    .children(rows.views())
                    .into_any_element()
            })
            .collect()
    }

    fn add_button(&self, list: List, label: &str, cx: &mut gpui::Context<Self>) -> AnyElement {
        let tag = if list == List::Layers { "layer" } else { "rule" };
        Button::new(SharedString::from(format!("{}-add-{tag}", self.id)))
            .icon(IconName::Plus)
            .label(label.to_string())
            .xsmall()
            .ghost()
            .on_click(cx.listener(move |this, _, window, cx| this.edit_list(list, ListEdit::Add, window, cx)))
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

/// A layer's or rule's title: its position and kind or material.
fn layer_title(index: usize, layer: &VoxelTerrainLayer) -> String {
    format!("{}. {:?}{}", index + 1, layer.kind, if layer.enabled { "" } else { " (off)" })
}

fn rule_title(index: usize, rule: &VoxelMaterialRule) -> String {
    format!("{}. {:?}", index + 1, rule.material)
}

impl gpui::Render for VoxelTerrainStackEditor {
    fn render(&mut self, _window: &mut Window, cx: &mut gpui::Context<Self>) -> impl gpui::IntoElement {
        let value = self.current.lock().map(|s| s.clone()).unwrap_or_default();
        let layer_titles = value.layers.iter().enumerate().map(|(i, l)| layer_title(i, l)).collect();
        let rule_titles = value.rules.iter().enumerate().map(|(i, r)| rule_title(i, r)).collect();
        let layers = self.list_items(List::Layers, layer_titles, cx);
        let rules = self.list_items(List::Rules, rule_titles, cx);
        let error = validation_error(&value).map(|error| {
            gpui::div().text_sm().text_color(cx.theme().danger).child(format!("Not generated: {error}"))
        });
        v_flex()
            .w_full()
            .gap_1()
            .child(pulsar_reflection::prims::editor_row(
                &self.label,
                Dropdown::new(&self.presets).xsmall().w(px(140.0)).placeholder("Apply preset"),
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
            .child(Self::heading("Overhangs", cx))
            .children(self.overhangs.views())
    }
}

fn voxel_terrain_stack_editor(args: &PropertyEditorArgs<'_>, window: &mut Window, cx: &mut App) -> BoundPropertyEditor {
    let entity = cx.new(|cx| VoxelTerrainStackEditor::new(args, window, cx));
    BoundPropertyEditor::new(entity, |editor: &mut VoxelTerrainStackEditor, value: &VoxelTerrainStack, window, cx| {
        editor.set_value(value, window, cx)
    })
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
        assert!(validation_error(&stack).is_some_and(|e| e.contains("Warp")), "{:?}", validation_error(&stack));
    }

    #[test]
    fn list_edits_move_remove_and_add() {
        let mut items = vec![1, 2, 3];
        apply_list_edit(&mut items, ListEdit::Up(1), || 0);
        assert_eq!(items, [2, 1, 3]);
        apply_list_edit(&mut items, ListEdit::Down(1), || 0);
        assert_eq!(items, [2, 3, 1]);
        apply_list_edit(&mut items, ListEdit::Up(0), || 0);
        apply_list_edit(&mut items, ListEdit::Down(2), || 0);
        assert_eq!(items, [2, 3, 1], "moves past the ends do nothing");
        apply_list_edit(&mut items, ListEdit::Remove(0), || 0);
        apply_list_edit(&mut items, ListEdit::Remove(9), || 0);
        assert_eq!(items, [3, 1]);
        apply_list_edit(&mut items, ListEdit::Add, || 7);
        assert_eq!(items, [3, 1, 7]);
    }
}
