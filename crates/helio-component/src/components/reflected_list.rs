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
        .child(button(
            "up",
            IconName::ArrowUp,
            "Move up",
            ListEdit::Up(index),
        ))
        .child(button(
            "down",
            IconName::ArrowDown,
            "Move down",
            ListEdit::Down(index),
        ))
        .child(button(
            "remove",
            IconName::Trash,
            "Remove",
            ListEdit::Remove(index),
        ));
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

/// Streaming fingerprint of a value's JSON: no allocation, so checking whether
/// a list changed costs a pass over it, not a copy of it.
fn fingerprint<T: serde::Serialize>(value: &[T]) -> u64 {
    use std::hash::Hasher as _;
    struct HashWriter(std::collections::hash_map::DefaultHasher);
    impl std::io::Write for HashWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.write(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = HashWriter(std::collections::hash_map::DefaultHasher::new());
    let _ = serde_json::to_writer(&mut writer, value);
    writer.0.finish()
}

/// Longer lists than this are drawn through a windowed list in a viewport of
/// their own, so only the cards in view exist as elements.
const WINDOWED_MIN_ITEMS: usize = 24;
const VIEWPORT_HEIGHT: f32 = 360.0;

/// A collapsible list item: a header (chevron, title, move and remove
/// buttons) and, when `expanded`, its rows.
///
/// `edit` applies a button's [`ListEdit`] to the owning view; `toggle` flips
/// the item's expansion.
pub(crate) fn collapsible_item_card<V: 'static>(
    id: &str,
    index: usize,
    title: String,
    expanded: bool,
    rows: impl IntoIterator<Item = AnyElement>,
    edit: impl Fn(&mut V, ListEdit, &mut Window, &mut gpui::Context<V>) + Clone + 'static,
    toggle: impl Fn(&mut V, usize, &mut gpui::Context<V>) + Clone + 'static,
    cx: &mut gpui::Context<V>,
) -> AnyElement {
    let mut button = |name: &str, icon: IconName, tooltip: &str, list_edit: ListEdit| {
        let edit = edit.clone();
        Button::new(SharedString::from(format!("{id}-{index}-{name}")))
            .icon(icon)
            .xsmall()
            .ghost()
            .tooltip(tooltip.to_string())
            .on_click(cx.listener(move |this, _, window, cx| {
                cx.stop_propagation();
                edit(this, list_edit, window, cx)
            }))
    };
    let buttons = h_flex()
        .gap_1()
        .child(button("up", IconName::ArrowUp, "Move up", ListEdit::Up(index)))
        .child(button(
            "down",
            IconName::ArrowDown,
            "Move down",
            ListEdit::Down(index),
        ))
        .child(button(
            "remove",
            IconName::Trash,
            "Remove",
            ListEdit::Remove(index),
        ));
    let toggle = toggle.clone();
    let header = h_flex()
        .id(SharedString::from(format!("{id}-{index}-header")))
        .w_full()
        .gap_1()
        .items_center()
        .cursor_pointer()
        .on_click(cx.listener(move |this, _, _, cx| toggle(this, index, cx)))
        .child(
            ui::Icon::new(if expanded {
                IconName::ChevronDown
            } else {
                IconName::ChevronRight
            })
            .xsmall()
            .text_color(cx.theme().muted_foreground),
        )
        .child(gpui::div().flex_1().text_sm().child(title))
        .child(buttons);
    v_flex()
        .w_full()
        .gap_1()
        .p_1()
        .rounded_md()
        .border_1()
        .border_color(cx.theme().border)
        .child(header)
        .when(expanded, |card| card.children(rows))
        .into_any_element()
}

/// A `Vec<T>` property as an ordered list of cards: move up and down,
/// remove, add, and every shown field of each item edited in place. The
/// whole list writes back as one value, so undo is one step.
///
/// Built to stay cheap on long lists of heavy items: cards start collapsed,
/// an item's field editors are created only when its card is first opened,
/// a change check is a streaming hash rather than a JSON copy, and a list
/// past [`WINDOWED_MIN_ITEMS`] is drawn through a windowed list.
pub struct ReflectedListEditor<T: ReflectedListItem> {
    label: String,
    id: String,
    current: Arc<Mutex<Vec<T>>>,
    write_back: PropertyWriteBack,
    /// Per item: its rows and the shown mask they were built for. `None`
    /// until the item's card is opened.
    items: Vec<Option<(FieldRows, Vec<bool>)>>,
    /// Opened cards, by index.
    expanded: std::collections::HashSet<usize>,
    /// Fingerprint of `current`, shared with the rows' write path so the
    /// value pushed back after a local edit is recognised as already shown.
    fingerprint: Arc<std::sync::atomic::AtomicU64>,
    list_state: gpui::ListState,
    scroll_state: ui::scroll::ScrollbarState,
    this: WeakEntity<Self>,
}

impl<T: ReflectedListItem> ReflectedListEditor<T> {
    pub fn new(
        args: &PropertyEditorArgs<'_>,
        _window: &mut Window,
        cx: &mut gpui::Context<Self>,
    ) -> Self {
        let value = args
            .current_value
            .downcast_ref::<Vec<T>>()
            .cloned()
            .unwrap_or_default();
        let hash = fingerprint(&value);
        let len = value.len();
        Self {
            label: args.display_name.to_string(),
            id: format!(
                "list-{}-{}-{}",
                args.id_prefix, args.class_name, args.prop_name
            ),
            current: Arc::new(Mutex::new(value)),
            write_back: args.write_back.clone(),
            items: (0..len).map(|_| None).collect(),
            expanded: Default::default(),
            fingerprint: Arc::new(std::sync::atomic::AtomicU64::new(hash)),
            list_state: gpui::ListState::new(len, gpui::ListAlignment::Top, gpui::px(300.0)),
            scroll_state: Default::default(),
            this: cx.weak_entity(),
        }
    }

    fn value(&self) -> Vec<T> {
        self.current.lock().map(|v| v.clone()).unwrap_or_default()
    }

    fn len(&self) -> usize {
        self.current.lock().map(|v| v.len()).unwrap_or(0)
    }

    /// Rows for item `i`, writing the whole list back on every edit.
    fn rows(&self, i: usize, item: &T, window: &mut Window, cx: &mut App) -> FieldRows {
        let (read, write) = (self.current.clone(), self.current.clone());
        let write_back = self.write_back.clone();
        let this = self.this.clone();
        let fingerprint_slot = self.fingerprint.clone();
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
                fingerprint_slot.store(
                    fingerprint(&items),
                    std::sync::atomic::Ordering::Release,
                );
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

    /// Build the rows of item `i` if it is open and they are missing or were
    /// built for a different shown mask.
    fn ensure_rows(&mut self, i: usize, window: &mut Window, cx: &mut App) {
        if !self.expanded.contains(&i) {
            return;
        }
        let Some(item) = self.current.lock().ok().and_then(|v| v.get(i).cloned()) else {
            return;
        };
        let mask = shown(&item);
        if self
            .items
            .get(i)
            .and_then(Option::as_ref)
            .is_some_and(|(_, built)| *built == mask)
        {
            return;
        }
        let rows = self.rows(i, &item, window, cx);
        if let Some(slot) = self.items.get_mut(i) {
            *slot = Some((rows, mask));
        }
    }

    /// Match the per-item slots to the list: a different length drops every
    /// built row (rows follow indices).
    fn sync_len(&mut self) {
        let len = self.len();
        if self.items.len() != len {
            self.items = (0..len).map(|_| None).collect();
            let old = self.list_state.item_count();
            self.list_state.splice(0..old, len);
        }
    }

    /// Push the current value into the rows that exist, and drop those whose
    /// shown fields changed (they rebuild when next drawn).
    fn refresh(&mut self, window: &mut Window, cx: &mut App) {
        self.sync_len();
        let value = self.value();
        for (i, item) in value.iter().enumerate() {
            let mask = shown(item);
            let stale = self
                .items
                .get(i)
                .and_then(Option::as_ref)
                .is_some_and(|(_, built)| *built != mask);
            if stale {
                self.items[i] = None;
            }
            if let Some(Some((rows, _))) = self.items.get(i) {
                rows.refresh(window, cx);
            }
        }
    }

    fn edit_list(&mut self, edit: ListEdit, window: &mut Window, cx: &mut gpui::Context<Self>) {
        let mut items = self.value();
        if edit == ListEdit::Add && items.len() >= T::MAX_ITEMS {
            return;
        }
        let before = items.len();
        apply_list_edit(&mut items, edit, T::default);
        self.expanded = remap_expanded(&self.expanded, edit, before);
        if edit == ListEdit::Add && items.len() > before {
            self.expanded.insert(items.len() - 1);
        }
        self.fingerprint
            .store(fingerprint(&items), std::sync::atomic::Ordering::Release);
        if let Ok(mut current) = self.current.lock() {
            *current = items.clone();
        }
        // Rows follow indices: every built item starts again.
        self.items = (0..items.len()).map(|_| None).collect();
        let old = self.list_state.item_count();
        self.list_state.splice(0..old, items.len());
        (self.write_back)(Box::new(items), window, cx);
        cx.notify();
    }

    fn toggle(&mut self, index: usize, cx: &mut gpui::Context<Self>) {
        if !self.expanded.remove(&index) {
            self.expanded.insert(index);
        }
        // The card changes height; the windowed list measures what it shows.
        self.list_state.remeasure();
        cx.notify();
    }

    /// Accept a value that changed elsewhere (undo, a preset, a blueprint).
    fn set_value(&mut self, value: &Vec<T>, window: &mut Window, cx: &mut gpui::Context<Self>) {
        let hash = fingerprint(value);
        if hash == self.fingerprint.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        self.fingerprint
            .store(hash, std::sync::atomic::Ordering::Release);
        if let Ok(mut current) = self.current.lock() {
            *current = value.clone();
        }
        self.refresh(window, cx);
        cx.notify();
    }

    /// One card: its header, and its rows when open.
    fn card(
        &mut self,
        i: usize,
        item: &T,
        window: &mut Window,
        cx: &mut gpui::Context<Self>,
    ) -> AnyElement {
        self.ensure_rows(i, window, cx);
        let open = self.expanded.contains(&i);
        let rows: Vec<AnyElement> = match self.items.get(i) {
            Some(Some((rows, _))) if open => rows.views().collect(),
            _ => Vec::new(),
        };
        collapsible_item_card(
            &self.id,
            i,
            item.title(i),
            open,
            rows,
            |this: &mut Self, edit, window, cx| this.edit_list(edit, window, cx),
            |this: &mut Self, index, cx| this.toggle(index, cx),
            cx,
        )
    }
}

/// Where the open cards sit after `edit` of a list that held `len` items.
fn remap_expanded(
    expanded: &std::collections::HashSet<usize>,
    edit: ListEdit,
    len: usize,
) -> std::collections::HashSet<usize> {
    expanded
        .iter()
        .filter_map(|&i| match edit {
            ListEdit::Up(m) if m > 0 && m < len => Some(if i == m {
                m - 1
            } else if i == m - 1 {
                m
            } else {
                i
            }),
            ListEdit::Down(m) if m + 1 < len => Some(if i == m {
                m + 1
            } else if i == m + 1 {
                m
            } else {
                i
            }),
            ListEdit::Remove(m) if m < len => match i.cmp(&m) {
                std::cmp::Ordering::Less => Some(i),
                std::cmp::Ordering::Equal => None,
                std::cmp::Ordering::Greater => Some(i - 1),
            },
            _ => Some(i),
        })
        .collect()
}

impl<T: ReflectedListItem> gpui::Render for ReflectedListEditor<T> {
    fn render(
        &mut self,
        window: &mut Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        self.sync_len();
        let len = self.len();
        let cards: Option<Vec<AnyElement>> = (len <= WINDOWED_MIN_ITEMS).then(|| {
            let value = self.value();
            value
                .iter()
                .enumerate()
                .map(|(i, item)| self.card(i, item, window, cx))
                .collect()
        });
        let windowed = cards.is_none().then(|| {
            let items = gpui::list(
                self.list_state.clone(),
                cx.processor(|this: &mut Self, i: usize, window, cx| {
                    let item = this.current.lock().ok().and_then(|v| v.get(i).cloned());
                    match item {
                        Some(item) => gpui::div()
                            .w_full()
                            .pb_1()
                            .child(this.card(i, &item, window, cx))
                            .into_any_element(),
                        None => gpui::div().into_any_element(),
                    }
                }),
            )
            .size_full();
            gpui::div()
                .id(SharedString::from(format!("{}-viewport", self.id)))
                .relative()
                .w_full()
                .h(gpui::px(VIEWPORT_HEIGHT))
                .child(items)
                .child(
                    gpui::div()
                        .absolute()
                        .top_0()
                        .right_0()
                        .bottom_0()
                        .w(gpui::px(12.))
                        .child(ui::scroll::Scrollbar::vertical(
                            &self.scroll_state,
                            &self.list_state,
                        )),
                )
        });
        let add = (len < T::MAX_ITEMS).then(|| {
            Button::new(SharedString::from(format!("{}-add", self.id)))
                .icon(IconName::Plus)
                .label(format!("Add {}", T::NOUN))
                .xsmall()
                .ghost()
                .on_click(
                    cx.listener(|this, _, window, cx| this.edit_list(ListEdit::Add, window, cx)),
                )
        });
        let count = if T::MAX_ITEMS == usize::MAX {
            format!("{} ({})", self.label, len)
        } else {
            format!("{} ({}/{})", self.label, len, T::MAX_ITEMS)
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
            .children(cards.into_iter().flatten())
            .children(windowed)
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
