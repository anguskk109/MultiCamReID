import numpy as np
from scipy.spatial.distance import cdist
import time

class GlobalReIDMatcher:
    def __init__(self, similarity_threshold=0.35, min_hits=2, temporal_window=5.0):
        self.similarity_threshold = similarity_threshold
        self.min_hits = min_hits
        self.temporal_window = temporal_window  # seconds
        self.global_id_counter = 0
        self.global_memory = []  # List of (id, feature)
        self.pending = []  # [(id, feature, hit_count)]
        self.recent_matches = []  # [(id, feature, timestamp)]
        self.local_to_global = {}  # Will be initialized externally per camera

    def match(self, features):
        global_ids = []
        used_ids = set()
        now = time.time()

        for feature in features:
            if not self.global_memory and not self.pending:
                # First ever person: assign ID = 1
                new_id = self._new_id()
                self.pending.append((new_id, feature, 1))
                global_ids.append(new_id)
                used_ids.add(new_id)
                self._update_recent(new_id, feature, now)
                continue  # move on to the next feature

            # First match against confirmed global memory
            if self.global_memory:
                existing_features = np.array([f for _, f in self.global_memory])
                distances = cdist([feature], existing_features, metric="cosine")[0]
                best_idx = np.argmin(distances)
                best_dist = distances[best_idx]

                if best_dist < self.similarity_threshold:
                    gid = self.global_memory[best_idx][0]
                    if gid in used_ids:
                        gid = None
                    else:
                        global_ids.append(gid)
                        used_ids.add(gid)
                        self._update_recent(gid, feature, now)
                        continue

            # First try recent confirmed IDs (temporal memory)
            gid = self._temporal_match(feature, now, used_ids)
            if gid is not None:
                global_ids.append(gid)
                self._update_recent(gid, feature, now)
                used_ids.add(gid)
            else:
                # Then try pending memory
                matched = False
                if self.pending:
                    pending_feats = np.array([f for _, f, _ in self.pending])
                    distances = cdist([feature], pending_feats, metric="cosine")[0]
                    best_idx = np.argmin(distances)
                    best_dist = distances[best_idx]

                    if best_dist < self.similarity_threshold:
                        pid, _, hits = self.pending[best_idx]
                        self.pending[best_idx] = (pid, feature, hits + 1)
                        if hits + 1 >= self.min_hits:
                            self.global_memory.append((pid, feature))
                            self.pending.pop(best_idx)
                        gid = pid
                        global_ids.append(gid)
                        used_ids.add(gid)
                        self._update_recent(gid, feature, now)
                        matched = True

                if not matched:
                    # Create new ID
                    new_id = self._new_id()
                    self.pending.append((new_id, feature, 1))
                    global_ids.append(new_id)
                    used_ids.add(new_id)
                    self._update_recent(new_id, feature, now)

        return global_ids

    def _temporal_match(self, feature, now, used_ids):
        if not self.recent_matches:
            return None
        recent_feats = np.array([f for _, f, t in self.recent_matches if now - t < self.temporal_window])
        if recent_feats.size == 0:
            return None
        distances = cdist([feature], recent_feats, metric="cosine")[0]
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]
        if best_dist < self.similarity_threshold:
            gid_candidates = [gid for gid, _, t in self.recent_matches if now - t < self.temporal_window]
            gid = gid_candidates[best_idx] if gid_candidates[best_idx] not in used_ids else None
            return gid
        return None

    def _update_recent(self, gid, feature, now):
        self.recent_matches.append((gid, feature, now))
        # Clean old entries
        self.recent_matches = [(g, f, t) for (g, f, t) in self.recent_matches if now - t < self.temporal_window]

    def update_memory(self, ids, features):
        # Ensure newest feature is updated in memory
        for gid, feat in zip(ids, features):
            for i, (stored_id, old_feat) in enumerate(self.global_memory):
                if stored_id == gid:
                    self.global_memory[i] = (gid, 0.9 * old_feat + 0.1 * feat)
                    break

    def _new_id(self):
        self.global_id_counter += 1
        return self.global_id_counter
