use std::cmp::Ordering;

/// Focus is independent of execution: refocusing a search keeps its results until an edit.
#[derive(Debug)]
pub struct SearchState {
    pub input: String,
    pub focused: bool,
    pub phase: SearchPhase,
}

#[derive(Debug)]
pub enum SearchPhase {
    Draft,
    Pending { request_id: u64 },
    Ready(SearchResults),
    Failed(String),
}

#[derive(Debug)]
pub struct SearchResults {
    embedding: Vec<f32>,
    matches: Vec<(usize, f32)>,
    cursor: SearchCursor,
}

#[derive(Clone, Copy, Debug)]
enum SearchCursor {
    AwaitingFirstMatch,
    Match(usize),
    /// The selected image was invalidated; remember the next match without selecting it.
    Gap(usize),
}

impl SearchState {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            focused: true,
            phase: SearchPhase::Draft,
        }
    }

    pub fn edit(&mut self, edit: impl FnOnce(&mut String)) {
        edit(&mut self.input);
        // Editing cancels acceptance of any outstanding response, including errors.
        self.phase = SearchPhase::Draft;
    }

    pub fn submit(&mut self, request_id: u64) {
        self.phase = SearchPhase::Pending { request_id };
        self.focused = false;
    }

    pub fn is_pending(&self, request_id: u64) -> bool {
        matches!(self.phase, SearchPhase::Pending { request_id: pending } if pending == request_id)
    }

    pub fn fail(&mut self, request_id: u64, error: String) {
        if self.is_pending(request_id) {
            self.phase = SearchPhase::Failed(error);
        }
    }

    pub fn complete(
        &mut self,
        request_id: u64,
        embedding: Vec<f32>,
        images: impl IntoIterator<Item = (usize, impl AsRef<[f32]>)>,
    ) -> Option<usize> {
        if !self.is_pending(request_id) {
            return None;
        }
        let mut matches: Vec<_> = images
            .into_iter()
            .map(|(index, image)| (index, similarity(&embedding, image.as_ref())))
            .collect();
        sort_matches(&mut matches);
        let first = matches.first().map(|(index, _)| *index);
        self.phase = SearchPhase::Ready(SearchResults {
            embedding,
            matches,
            cursor: if first.is_some() {
                SearchCursor::Match(0)
            } else {
                SearchCursor::AwaitingFirstMatch
            },
        });
        first
    }

    pub fn results(&self) -> &[(usize, f32)] {
        match &self.phase {
            SearchPhase::Ready(results) => &results.matches,
            _ => &[],
        }
    }

    pub fn current(&self) -> Option<usize> {
        match &self.phase {
            SearchPhase::Ready(SearchResults {
                cursor: SearchCursor::Match(current),
                ..
            }) => Some(*current),
            _ => None,
        }
    }

    pub fn advance(&mut self, delta: isize) -> Option<usize> {
        let SearchPhase::Ready(results) = &mut self.phase else {
            return None;
        };
        if results.matches.is_empty() {
            return None;
        }
        let position = match results.cursor {
            SearchCursor::Match(current) => current as isize,
            SearchCursor::Gap(next) if delta != 0 => next as isize - isize::from(delta > 0),
            _ => return None,
        };
        let current = (position + delta).rem_euclid(results.matches.len() as isize) as usize;
        results.cursor = SearchCursor::Match(current);
        Some(results.matches[current].0)
    }

    /// Keep the selected image stable while background indexing reorders the matches.
    pub fn image_ready(&mut self, index: usize, image: &[f32], selected: usize) -> Option<usize> {
        let SearchPhase::Ready(results) = &mut self.phase else {
            return None;
        };
        let anchor = match results.cursor {
            SearchCursor::Match(position) | SearchCursor::Gap(position) => {
                results.matches.get(position).map(|(index, _)| *index)
            }
            SearchCursor::AwaitingFirstMatch => None,
        };
        let select_first = matches!(results.cursor, SearchCursor::AwaitingFirstMatch);
        let score = similarity(&results.embedding, image);
        if let Some(entry) = results.matches.iter_mut().find(|(idx, _)| *idx == index) {
            entry.1 = score;
        } else {
            results.matches.push((index, score));
        }
        sort_matches(&mut results.matches);
        let selected_position = results.matches.iter().position(|(idx, _)| *idx == selected);
        let anchor_position =
            anchor.and_then(|anchor| results.matches.iter().position(|(idx, _)| *idx == anchor));
        results.cursor = if let Some(position) = selected_position {
            SearchCursor::Match(position)
        } else {
            match results.cursor {
                SearchCursor::AwaitingFirstMatch => SearchCursor::Match(0),
                SearchCursor::Match(_) => SearchCursor::Match(anchor_position.unwrap()),
                SearchCursor::Gap(_) => {
                    SearchCursor::Gap(anchor_position.unwrap_or(results.matches.len()))
                }
            }
        };
        select_first.then_some(index)
    }

    pub fn invalidate_image(&mut self, index: usize) {
        if let SearchPhase::Ready(results) = &mut self.phase {
            let Some(removed) = results.matches.iter().position(|(idx, _)| *idx == index) else {
                return;
            };
            results.matches.remove(removed);
            results.cursor = match results.cursor {
                SearchCursor::Match(current) if current == removed => SearchCursor::Gap(current),
                SearchCursor::Match(current) => {
                    SearchCursor::Match(current - usize::from(removed < current))
                }
                SearchCursor::Gap(next) => SearchCursor::Gap(next - usize::from(removed < next)),
                SearchCursor::AwaitingFirstMatch => SearchCursor::AwaitingFirstMatch,
            };
        }
    }
}

fn similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn sort_matches(matches: &mut [(usize, f32)]) {
    matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_and_resubmission_reject_late_success_and_failure() {
        let mut search = SearchState::new();
        search.submit(1);
        search.focused = true;
        search.edit(|input| input.push('猫'));
        search.complete(1, vec![1.0], [(0, [1.0])]);
        search.fail(1, "old error".into());
        assert!(matches!(search.phase, SearchPhase::Draft));
        assert_eq!(search.input, "猫");

        search.submit(2);
        search.complete(1, vec![1.0], [(0, [1.0])]);
        search.fail(1, "old error".into());
        assert!(search.is_pending(2));
        assert_eq!(search.complete(2, vec![1.0], [(1, [1.0])]), Some(1));
        search.fail(2, "duplicate error".into());
        assert_eq!(search.results(), &[(1, 1.0)]);
    }

    #[test]
    fn indexing_can_populate_empty_results_and_preserves_selected_image() {
        let mut search = SearchState::new();
        search.submit(0);
        search.complete(0, vec![1.0], Vec::<(usize, Vec<f32>)>::new());
        assert_eq!(search.advance(1), None);
        assert_eq!(search.image_ready(0, &[0.5], 0), Some(0));
        assert_eq!(search.image_ready(1, &[0.9], 0), None);
        assert_eq!(search.current(), Some(1));
        assert_eq!(search.advance(1), Some(1));
        assert_eq!(search.advance(-1), Some(0));
        search.invalidate_image(1);
        assert_eq!(search.results(), &[(0, 0.5)]);
        assert_eq!(search.current(), Some(0));
        search.invalidate_image(0);
        assert_eq!(search.advance(-1), None);
        assert_eq!(search.current(), None);
    }

    #[test]
    fn invalidating_the_selected_match_preserves_both_navigation_directions() {
        for selected in 0..3 {
            for direction in [-1, 1] {
                let mut search = SearchState::new();
                search.submit(0);
                search.complete(0, vec![1.0], [(0, [0.9]), (1, [0.8]), (2, [0.7])]);
                assert_eq!(search.advance(selected as isize), Some(selected));

                search.invalidate_image(selected);
                assert_eq!(search.current(), None);
                assert_eq!(search.advance(0), None);
                let expected = (selected as isize + direction).rem_euclid(3) as usize;
                assert_eq!(search.advance(direction), Some(expected));
            }
        }
    }

    #[test]
    fn indexing_and_invalidation_preserve_the_next_match() {
        let mut search = SearchState::new();
        search.submit(0);
        search.complete(0, vec![1.0], [(0, [0.9]), (1, [0.8]), (2, [0.7])]);
        search.invalidate_image(0);

        // New higher scores must not displace the next match after the invalidated image.
        assert_eq!(search.image_ready(3, &[1.0], 0), None);
        assert_eq!(search.current(), None);
        assert_eq!(search.advance(1), Some(1));

        search.invalidate_image(1);
        search.invalidate_image(3);
        assert_eq!(search.advance(1), Some(2));
    }

    #[test]
    fn invalidating_the_only_match_does_not_autoselect_another_image() {
        let mut search = SearchState::new();
        search.submit(0);
        search.complete(0, vec![1.0], [(0, [0.9])]);
        search.invalidate_image(0);

        assert_eq!(search.image_ready(1, &[0.8], 0), None);
        assert_eq!(search.current(), None);
        assert_eq!(search.image_ready(0, &[0.7], 0), None);
        assert_eq!(search.current(), Some(1));
        assert_eq!(search.advance(1), Some(1));
    }
}
