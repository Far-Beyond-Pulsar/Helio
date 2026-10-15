//! Inspector editing for ordered lists of reflected values.
//!
//! The reflected properties panel finds an editor by the property's exact
//! type, and nothing registers one for a `Vec<T>`, so a list field shows as
//! unimplemented. This module holds the pieces a list editor is built from:
//! [`FieldRows`] (every field of one value, each edited by the editor
//! registered for its type), [`ListEdit`] (move, remove, add) and
//! [`item_card`] (one item's header and rows). [`ReflectedListEditor`]
//! puts them together for any [`ReflectedListItem`]; register it for a
//! `Vec<T>` with [`register_reflected_list_editor!`].

use std::any::Any;
use std::sync::{Arc, Mutex};

use gpui::{AnyElement, App, SharedString, WeakEntity, Window, prelude::*};
use pulsar_reflection::{
    BoundPropertyEditor, EngineClass, PropertyEditorArgs, PropertyEditorFactory, PropertyMetadata,
    PropertyWriteBack,
};
use ui::button::{Button, ButtonVariants as _};
use ui::{ActiveTheme as _, IconName, Sizable as _, h_flex, v_flex};

/// The editor registered for a property type, if any.
pub(crate) fn editor_factory(type_id: std::any::TypeId) -> Option<PropertyEditorFactory> {
    pulsar_reflection::inventory::iter::<pulsar_reflection::UiPropertyEditorHint>
        .into_iter()
        .find(|hint| hint.type_id == type_id)
        // SAFETY: `erase_property_editor_fn_ptr` only accepts a
        // `PropertyEditorFactory`, so every submitted pointer has that type
        // (the same contract the inspector's registry relies on).
        .map(|hint| unsafe { std::mem::transmute::<fn(), PropertyEditorFactory>(hint.fn_ptr) })
}

/// Editors for the reflected properties of one value (a layer, a rule, a
/// lens element...), each writing the whole value back through `put`.
/// Properties without a registered editor (lists, nested structs) and those
/// `shows` hides are skipped.
pub(crate) struct FieldRows {
    editors: Vec<(usize, BoundPropertyEditor)>,
    /// Pushes the current value into every editor.
    refresh: Arc<dyn Fn(&[(usize, BoundPropertyEditor)], &mut Window, &mut App) + Send + Sync>,
}

impl FieldRows {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<T: EngineClass + Clone>(
        value: &T,
        id: &str,
        class_name: &str,
        shows: impl Fn(&T, &PropertyMetadata) -> bool,
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
            if !shows(value, prop) {
                continue;
            }
            let Some(factory) = editor_factory(prop.type_info.type_id) else {
                continue;
            };
            let current = (prop.getter)(value);
            let write_back: PropertyWriteBack = {
                let (props, get, put) = (props.clone(), get.clone(), put.clone());
                Arc::new(
                    move |new: Box<dyn Any + Send>, window: &mut Window, cx: &mut App| {
                        let Some(mut value) = get() else { return };
                        (props[index].setter)(&mut value, new);
                        put(value, window, cx);
                    },
                )
            };
            let args = PropertyEditorArgs {
                id_prefix: id,
                class_name,
                display_name: &prop.display_name,
                prop_name: prop.name,
                type_info: prop.type_info,
                current_value: current.as_ref(),
                write_back,
            };
            editors.push((index, factory(&args, window, cx)));
        }
        let refresh: Arc<
            dyn Fn(&[(usize, BoundPropertyEditor)], &mut Window, &mut App) + Send + Sync,
        > = Arc::new(
            move |editors: &[(usize, BoundPropertyEditor)], window: &mut Window, cx: &mut App| {
                let Some(value) = get() else { return };
                for (index, editor) in editors {
                    (editor.set_value)((props[*index].getter)(&value).as_ref(), window, cx);
                }
            },
        );
        Self { editors, refresh }
    }

    pub(crate) fn refresh(&self, window: &mut Window, cx: &mut App) {
        (self.refresh)(&self.editors, window, cx);
    }

    pub(crate) fn views(&self) -> impl Iterator<Item = AnyElement> + '_ {
        self.editors
            .iter()
            .map(|(_, editor)| editor.view.clone().into_any_element())
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

/// One list item: its title with move and remove buttons, then its rows.
/// `edit` applies a button's [`ListEdit`] to the owning view.
pub(crate) fn item_card<V: 'static>(
    id: &str,
    index: usize,
    title: String,
    rows: impl IntoIterator<Item = AnyElement>,
    edit: impl Fn(&mut V, ListEdit, &mut Window, &mut gpui::Context<V>) + Clone + 'static,
    cx: &mut gpui::Context<V>,
) -> AnyElement {
    let mut button = |name: &str, icon: IconName, tooltip: &str, list_edit: ListEdit| {
        let edit = edit.clone();
        Button::new(SharedString::from(format!("{id}-{index}-{name}")))
            .icon(icon)
            .xsmall()
            .ghost()
            .tooltip(tooltip.to_string())
            .on_click(cx.listener(move |this, _, window, cx| edit(this, list_edit, window, cx)))
    };
    let buttons = h_flex()
        .gap_1()
        .child(button("up", IconName::ArrowUp, "Move up", ListEdit::Up(index)))
        .child(button("down", IconName::ArrowDown, "Move down", ListEdit::Down(index)))
        .child(button("remove", IconName::Trash, "Remove", ListEdit::Remove(index)));
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
                .child(buttons),
        )
        .children(rows)
        .into_any_element()
}

/// A value edited as an item of a `Vec` property by [`ReflectedListEditor`].
pub trait ReflectedListItem:
    EngineClass + Clone + Default + serde::Serialize + Send + Sync + 'static
{
    /// Class name handed to the field editors.
    const CLASS_NAME: &'static str;
    /// What one item is called on the add button ("element").
    const NOUN: &'static str;
    /// Most items the list holds; the add button hides at this length.
    const MAX_ITEMS: usize = usize::MAX;

    /// The item's card title; `index` is its position.
    fn title(&self, index: usize) -> String;

    /// Whether the item's card shows `prop`. An item whose fields mean
    /// different things by kind shows only those its kind reads; its rows
    /// rebuild when the answer changes.
    fn shows(&self, _prop: &PropertyMetadata) -> bool {
        true
    }

    /// Store an edited item. Returns true when more than the edited field
    /// changed (a new kind bringing its defaults), so every row refreshes.
    fn apply_edit(&mut self, edited: Self) -> bool {
        *self = edited;
        false
    }
}

/// Which of an item's properties its card shows.
fn shown<T: ReflectedListItem>(item: &T) -> Vec<bool> {
    item.get_properties()
        .iter()
        .map(|prop| item.shows(prop))
        .collect()
}

/// A `Vec<T>` property as an ordered list of cards: move up and down,
/// remove, add, and every shown field of each item edited in place. The
/// whole list writes back as one value, so undo is one step.
pub struct ReflectedListEditor<T: ReflectedListItem> {
    label: String,
    id: String,
    current: Arc<Mutex<Vec<T>>>,
    write_back: PropertyWriteBack,
    /// Per item: its rows and the shown mask they were built for.
    items: Vec<(FieldRows, Vec<bool>)>,
    this: WeakEntity<Self>,
}

impl<T: ReflectedListItem> ReflectedListEditor<T> {
    pub fn new(args: &PropertyEditorArgs<'_>, window: &mut Window, cx: &mut gpui::Context<Self>) -> Self {
        let value = args
            .current_value
            .downcast_ref::<Vec<T>>()
            .cloned()
            .unwrap_or_default();
        let mut editor = Self {
            label: args.display_name.to_string(),
            id: format!("list-{}-{}-{}", args.id_prefix, args.class_name, args.prop_name),
            current: Arc::new(Mutex::new(value)),
            write_back: args.write_back.clone(),
            items: Vec::new(),
            this: cx.weak_entity(),
        };
        editor.sync_rows(window, cx);
        editor
    }

    fn value(&self) -> Vec<T> {
        self.current.lock().map(|v| v.clone()).unwrap_or_default()
    }

    /// Rows for item `i`, writing the whole list back on every edit.
    fn rows(&self, i: usize, item: &T, window: &mut Window, cx: &mut App) -> FieldRows {
        let (read, write) = (self.current.clone(), self.current.clone());
        let write_back = self.write_back.clone();
        let this = self.this.clone();
        FieldRows::new(
            item,
            &format!("{}-{i}", self.id),
            T::CLASS_NAME,
            T::shows,
            move || read.lock().ok().and_then(|items| items.get(i).cloned()),
            move |edited: T, window, cx| {
                let (items, reshaped) = {
                    let Ok(mut items) = write.lock() else { return };
                    let Some(item) = items.get_mut(i) else { return };
                    let before = shown(item);
                    let refreshed = item.apply_edit(edited);
                    let reshaped = refreshed || shown(item) != before;
                    (items.clone(), reshaped)
                };
                write_back(Box::new(items), window, cx);
                if reshaped {
                    // After the editor that wrote returns: it is mid-event,
                    // and its own row may be the one rebuilt.
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

    /// Rebuild the rows of every item whose shown fields changed, and all of
    /// them when the length changed (rows follow indices).
    fn sync_rows(&mut self, window: &mut Window, cx: &mut App) {
        let value = self.value();
        if self.items.len() != value.len() {
            self.items.clear();
        }
        for (i, item) in value.iter().enumerate() {
            let mask = shown(item);
            if self.items.get(i).is_some_and(|(_, built)| *built == mask) {
                continue;
            }
            let rows = self.rows(i, item, window, cx);
            if i < self.items.len() {
                self.items[i] = (rows, mask);
            } else {
                self.items.push((rows, mask));
            }
        }
    }

    fn refresh(&mut self, window: &mut Window, cx: &mut App) {
        self.sync_rows(window, cx);
        for (rows, _) in &self.items {
            rows.refresh(window, cx);
        }
    }

    fn edit_list(&mut self, edit: ListEdit, window: &mut Window, cx: &mut gpui::Context<Self>) {
        let mut items = self.value();
        if edit == ListEdit::Add && items.len() >= T::MAX_ITEMS {
            return;
        }
        apply_list_edit(&mut items, edit, T::default);
        if let Ok(mut current) = self.current.lock() {
            *current = items.clone();
        }
        self.refresh(window, cx);
        (self.write_back)(Box::new(items), window, cx);
        cx.notify();
    }

    /// Accept a value that changed elsewhere (undo, a preset, a blueprint).
    fn set_value(&mut self, value: &Vec<T>, window: &mut Window, cx: &mut gpui::Context<Self>) {
        let changed = self
            .current
            .lock()
            .map(|current| serde_json::to_value(&*current).ok() != serde_json::to_value(value).ok())
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
}

impl<T: ReflectedListItem> gpui::Render for ReflectedListEditor<T> {
    fn render(&mut self, _window: &mut Window, cx: &mut gpui::Context<Self>) -> impl gpui::IntoElement {
        let value = self.value();
        let cards: Vec<AnyElement> = self
            .items
            .iter()
            .zip(&value)
            .enumerate()
            .map(|(i, ((rows, _), item))| {
                item_card(
                    &self.id,
                    i,
                    item.title(i),
                    rows.views(),
                    |this: &mut Self, edit, window, cx| this.edit_list(edit, window, cx),
                    cx,
                )
            })
            .collect();
        let add = (value.len() < T::MAX_ITEMS).then(|| {
            Button::new(SharedString::from(format!("{}-add", self.id)))
                .icon(IconName::Plus)
                .label(format!("Add {}", T::NOUN))
                .xsmall()
                .ghost()
                .on_click(cx.listener(|this, _, window, cx| {
                    this.edit_list(ListEdit::Add, window, cx)
                }))
        });
        let count = if T::MAX_ITEMS == usize::MAX {
            format!("{} ({})", self.label, value.len())
        } else {
            format!("{} ({}/{})", self.label, value.len(), T::MAX_ITEMS)
        };
        v_flex()
            .w_full()
            .gap_1()
            .child(
                gpui::div()
                    .text_sm()
                    .text_color(cx.theme().foreground)
                    .child(count),
            )
            .children(cards)
            .children(add)
    }
}

/// The editor factory for a `Vec<T>` property.
pub fn reflected_list_editor<T: ReflectedListItem>(
    args: &PropertyEditorArgs<'_>,
    window: &mut Window,
    cx: &mut App,
) -> BoundPropertyEditor {
    let entity = cx.new(|cx| ReflectedListEditor::<T>::new(args, window, cx));
    BoundPropertyEditor::new(
        entity,
        |editor: &mut ReflectedListEditor<T>, value: &Vec<T>, window, cx| {
            editor.set_value(value, window, cx)
        },
    )
}

/// Register [`ReflectedListEditor`] as the inspector editor of `Vec<$item>`.
#[macro_export]
macro_rules! register_reflected_list_editor {
    ($item:ty) => {
        pulsar_reflection::inventory::submit! {
            pulsar_reflection::UiPropertyEditorHint {
                type_id: std::any::TypeId::of::<Vec<$item>>(),
                fn_ptr: pulsar_reflection::erase_property_editor_fn_ptr(
                    $crate::components::reflected_list::reflected_list_editor::<$item>,
                ),
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

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
