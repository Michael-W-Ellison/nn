// File: src/storage/persistent_backend.cpp
#include "storage/persistent_backend.hpp"
#include <sstream>
#include <fstream>
#include <cstring>
#include <sys/stat.h>

namespace dpan {

// ============================================================================
// Constructor and Destructor
// ============================================================================

PersistentBackend::PersistentBackend(const Config& config)
    : config_(config) {

    // Open SQLite database
    int rc = sqlite3_open(config_.db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::string error = sqlite3_errmsg(db_);
        sqlite3_close(db_);
        throw std::runtime_error("Failed to open database: " + error);
    }

    // Initialize database schema and settings
    InitializeDatabase();
}

PersistentBackend::~PersistentBackend() {
    if (db_) {
        // Use sqlite3_close_v2() instead of sqlite3_close()
        // This properly handles WAL checkpointing and waits for all statements to finish
        // Prevents hanging when destructor is called with active transactions
        int rc = sqlite3_close_v2(db_);
        if (rc != SQLITE_OK) {
            // Log error but don't throw in destructor
            // The database will be closed eventually when all statements are finalized
        }
        db_ = nullptr;
    }
}

// ============================================================================
// Database Initialization
// ============================================================================

void PersistentBackend::InitializeDatabase() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Set busy timeout to prevent infinite waiting on locks (5 seconds)
    // This is CRITICAL to prevent tests from hanging
    sqlite3_busy_timeout(db_, 5000);

    // Set pragmas for performance
    if (config_.enable_wal) {
        ExecuteSQL("PRAGMA journal_mode=WAL;");
    }

    ExecuteSQL("PRAGMA synchronous=" + config_.synchronous + ";");
    ExecuteSQL("PRAGMA cache_size=-" + std::to_string(config_.cache_size_kb) + ";");
    ExecuteSQL("PRAGMA page_size=" + std::to_string(config_.page_size) + ";");

    if (config_.enable_auto_vacuum) {
        ExecuteSQL("PRAGMA auto_vacuum=INCREMENTAL;");
    }

    // Create tables if they don't exist
    CreateTables();

    // Create indices for efficient queries
    CreateIndices();
}

void PersistentBackend::CreateTables() {
    // Main patterns table
    std::string create_table = R"(
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY,
            type INTEGER NOT NULL,
            creation_time INTEGER NOT NULL,
            data BLOB NOT NULL
        );
    )";

    if (!ExecuteSQL(create_table)) {
        throw std::runtime_error("Failed to create patterns table");
    }
}

void PersistentBackend::CreateIndices() {
    // Index on type for FindByType queries
    ExecuteSQL("CREATE INDEX IF NOT EXISTS idx_type ON patterns(type);");

    // Index on creation_time for FindByTimeRange queries
    ExecuteSQL("CREATE INDEX IF NOT EXISTS idx_creation_time ON patterns(creation_time);");
}

bool PersistentBackend::ExecuteSQL(const std::string& sql) {
    char* error_msg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &error_msg);

    if (rc != SQLITE_OK) {
        if (error_msg) {
            sqlite3_free(error_msg);
        }
        return false;
    }

    return true;
}

// ============================================================================
// Core CRUD Operations
// ============================================================================

bool PersistentBackend::Store(const PatternNode& node) {
    std::lock_guard<std::mutex> lock(mutex_);

    total_writes_.fetch_add(1, std::memory_order_relaxed);

    // Prepare INSERT statement
    const char* sql = "INSERT INTO patterns (id, type, creation_time, data) VALUES (?, ?, ?, ?);";
    sqlite3_stmt* stmt;

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    // Bind parameters
    sqlite3_bind_int64(stmt, 1, node.GetID().value());
    sqlite3_bind_int(stmt, 2, static_cast<int>(node.GetType()));
    sqlite3_bind_int64(stmt, 3, node.GetCreationTime().ToMicros());

    // Serialize pattern data
    std::vector<uint8_t> blob = SerializeNode(node);
    sqlite3_bind_blob(stmt, 4, blob.data(), blob.size(), SQLITE_TRANSIENT);

    // Execute
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return rc == SQLITE_DONE;
}

std::optional<PatternNode> PersistentBackend::Retrieve(PatternID id) {
    std::lock_guard<std::mutex> lock(mutex_);

    total_reads_.fetch_add(1, std::memory_order_relaxed);

    // Prepare SELECT statement
    const char* sql = "SELECT data FROM patterns WHERE id = ?;";
    sqlite3_stmt* stmt;

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return std::nullopt;
    }

    // Bind parameter
    sqlite3_bind_int64(stmt, 1, id.value());

    // Execute
    rc = sqlite3_step(stmt);

    if (rc == SQLITE_ROW) {
        // Read blob
        const void* blob_data = sqlite3_column_blob(stmt, 0);
        int blob_size = sqlite3_column_bytes(stmt, 0);

        std::vector<uint8_t> blob(static_cast<const uint8_t*>(blob_data),
                                  static_cast<const uint8_t*>(blob_data) + blob_size);

        sqlite3_finalize(stmt);

        // Deserialize
        return DeserializeNode(blob);
    }

    sqlite3_finalize(stmt);
    return std::nullopt;
}

bool PersistentBackend::Update(const PatternNode& node) {
    std::lock_guard<std::mutex> lock(mutex_);

    total_writes_.fetch_add(1, std::memory_order_relaxed);

    // Prepare UPDATE statement
    const char* sql = "UPDATE patterns SET type = ?, creation_time = ?, data = ? WHERE id = ?;";
    sqlite3_stmt* stmt;

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    // Bind parameters
    sqlite3_bind_int(stmt, 1, static_cast<int>(node.GetType()));
    sqlite3_bind_int64(stmt, 2, node.GetCreationTime().ToMicros());

    std::vector<uint8_t> blob = SerializeNode(node);
    sqlite3_bind_blob(stmt, 3, blob.data(), blob.size(), SQLITE_TRANSIENT);

    sqlite3_bind_int64(stmt, 4, node.GetID().value());

    // Execute
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        return false;
    }

    // Check if any row was updated
    return sqlite3_changes(db_) > 0;
}

bool PersistentBackend::Delete(PatternID id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Prepare DELETE statement
    const char* sql = "DELETE FROM patterns WHERE id = ?;";
    sqlite3_stmt* stmt;

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    // Bind parameter
    sqlite3_bind_int64(stmt, 1, id.value());

    // Execute
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        return false;
    }

    // Check if any row was deleted
    return sqlite3_changes(db_) > 0;
}

bool PersistentBackend::Exists(PatternID id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Prepare SELECT statement
    const char* sql = "SELECT 1 FROM patterns WHERE id = ? LIMIT 1;";
    sqlite3_stmt* stmt;

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    // Bind parameter
    sqlite3_bind_int64(stmt, 1, id.value());

    // Execute
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return rc == SQLITE_ROW;
}

// ============================================================================
// Batch Operations
// ============================================================================

size_t PersistentBackend::StoreBatch(const std::vector<PatternNode>& nodes) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (nodes.empty()) {
        return 0;
    }

    // Begin transaction
    BeginTransaction();

    size_t stored_count = 0;

    // Prepare statement once, execute multiple times
    const char* sql = "INSERT OR IGNORE INTO patterns (id, type, creation_time, data) VALUES (?, ?, ?, ?);";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        RollbackTransaction();
        return 0;
    }

    for (const auto& node : nodes) {
        // Bind parameters
        sqlite3_bind_int64(stmt, 1, node.GetID().value());
        sqlite3_bind_int(stmt, 2, static_cast<int>(node.GetType()));
        sqlite3_bind_int64(stmt, 3, node.GetCreationTime().ToMicros());

        std::vector<uint8_t> blob = SerializeNode(node);
        sqlite3_bind_blob(stmt, 4, blob.data(), blob.size(), SQLITE_TRANSIENT);

        // Execute
        if (sqlite3_step(stmt) == SQLITE_DONE) {
            if (sqlite3_changes(db_) > 0) {
                ++stored_count;
            }
        }

        // Reset statement for next iteration
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);
    CommitTransaction();

    total_writes_.fetch_add(stored_count, std::memory_order_relaxed);

    return stored_count;
}

std::vector<PatternNode> PersistentBackend::RetrieveBatch(const std::vector<PatternID>& ids) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<PatternNode> results;
    results.reserve(ids.size());

    // Prepare SELECT statement
    const char* sql = "SELECT data FROM patterns WHERE id = ?;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return results;
    }

    for (const auto& id : ids) {
        sqlite3_bind_int64(stmt, 1, id.value());

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const void* blob_data = sqlite3_column_blob(stmt, 0);
            int blob_size = sqlite3_column_bytes(stmt, 0);

            std::vector<uint8_t> blob(static_cast<const uint8_t*>(blob_data),
                                      static_cast<const uint8_t*>(blob_data) + blob_size);

            results.push_back(DeserializeNode(blob));
        }

        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);

    total_reads_.fetch_add(results.size(), std::memory_order_relaxed);

    return results;
}

size_t PersistentBackend::DeleteBatch(const std::vector<PatternID>& ids) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (ids.empty()) {
        return 0;
    }

    BeginTransaction();

    size_t deleted_count = 0;

    const char* sql = "DELETE FROM patterns WHERE id = ?;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        RollbackTransaction();
        return 0;
    }

    for (const auto& id : ids) {
        sqlite3_bind_int64(stmt, 1, id.value());

        if (sqlite3_step(stmt) == SQLITE_DONE) {
            deleted_count += sqlite3_changes(db_);
        }

        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);
    CommitTransaction();

    return deleted_count;
}

// ============================================================================
// Query Operations
// ============================================================================

std::vector<PatternID> PersistentBackend::FindByType(
        PatternType type,
        const QueryOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<PatternID> results;

    const char* sql = "SELECT id FROM patterns WHERE type = ? LIMIT ?;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return results;
    }

    sqlite3_bind_int(stmt, 1, static_cast<int>(type));
    sqlite3_bind_int64(stmt, 2, options.max_results);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        uint64_t id_value = sqlite3_column_int64(stmt, 0);
        results.push_back(PatternID(id_value));
    }

    sqlite3_finalize(stmt);

    return results;
}

std::vector<PatternID> PersistentBackend::FindByTimeRange(
        Timestamp start,
        Timestamp end,
        const QueryOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<PatternID> results;

    const char* sql = "SELECT id FROM patterns WHERE creation_time >= ? AND creation_time <= ? LIMIT ?;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return results;
    }

    sqlite3_bind_int64(stmt, 1, start.ToMicros());
    sqlite3_bind_int64(stmt, 2, end.ToMicros());
    sqlite3_bind_int64(stmt, 3, options.max_results);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        uint64_t id_value = sqlite3_column_int64(stmt, 0);
        results.push_back(PatternID(id_value));
    }

    sqlite3_finalize(stmt);

    return results;
}

std::vector<PatternID> PersistentBackend::FindAll(const QueryOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<PatternID> results;

    const char* sql = "SELECT id FROM patterns LIMIT ?;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return results;
    }

    sqlite3_bind_int64(stmt, 1, options.max_results);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        uint64_t id_value = sqlite3_column_int64(stmt, 0);
        results.push_back(PatternID(id_value));
    }

    sqlite3_finalize(stmt);

    return results;
}

// ============================================================================
// Statistics and Monitoring
// ============================================================================

// Internal helper - assumes mutex is already locked
size_t PersistentBackend::CountUnlocked() const {
    const char* sql = "SELECT COUNT(*) FROM patterns;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return 0;
    }

    size_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }

    sqlite3_finalize(stmt);

    return count;
}

size_t PersistentBackend::Count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return CountUnlocked();
}

StorageStats PersistentBackend::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    StorageStats stats;
    stats.total_patterns = CountUnlocked();  // Use unlocked version - mutex already held
    stats.disk_usage_bytes = GetDatabaseSize();
    stats.memory_usage_bytes = 0;  // SQLite manages its own cache

    // Calculate average read/write times (simplified)
    uint64_t total_ops = total_reads_.load() + total_writes_.load();
    if (total_ops > 0) {
        stats.avg_lookup_time_ms = 1.0f;  // Placeholder
    }

    return stats;
}

// ============================================================================
// Maintenance Operations
// ============================================================================

void PersistentBackend::Flush() {
    std::lock_guard<std::mutex> lock(mutex_);

    // WAL checkpoint
    if (config_.enable_wal) {
        ExecuteSQL("PRAGMA wal_checkpoint(FULL);");
    }
}

void PersistentBackend::Compact() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Run VACUUM to reclaim space
    ExecuteSQL("VACUUM;");

    // Incremental vacuum if auto-vacuum is enabled
    if (config_.enable_auto_vacuum) {
        ExecuteSQL("PRAGMA incremental_vacuum;");
    }
}

void PersistentBackend::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    ExecuteSQL("DELETE FROM patterns;");

    // Reset statistics
    total_reads_.store(0, std::memory_order_relaxed);
    total_writes_.store(0, std::memory_order_relaxed);
}

// ============================================================================
// Snapshot and Restore
// ============================================================================

bool PersistentBackend::CreateSnapshot(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Flush WAL first
    if (config_.enable_wal) {
        ExecuteSQL("PRAGMA wal_checkpoint(FULL);");
    }

    // Use SQLite backup API
    sqlite3* backup_db;
    int rc = sqlite3_open(path.c_str(), &backup_db);
    if (rc != SQLITE_OK) {
        return false;
    }

    sqlite3_backup* backup = sqlite3_backup_init(backup_db, "main", db_, "main");
    if (!backup) {
        sqlite3_close(backup_db);
        return false;
    }

    sqlite3_backup_step(backup, -1);  // Copy all pages
    sqlite3_backup_finish(backup);

    rc = sqlite3_errcode(backup_db);
    sqlite3_close(backup_db);

    return rc == SQLITE_OK;
}

bool PersistentBackend::RestoreSnapshot(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Open backup database
    sqlite3* backup_db;
    int rc = sqlite3_open(path.c_str(), &backup_db);
    if (rc != SQLITE_OK) {
        return false;
    }

    // Clear current database
    ExecuteSQL("DELETE FROM patterns;");

    // Use backup API to restore
    sqlite3_backup* backup = sqlite3_backup_init(db_, "main", backup_db, "main");
    if (!backup) {
        sqlite3_close(backup_db);
        return false;
    }

    sqlite3_backup_step(backup, -1);  // Copy all pages
    sqlite3_backup_finish(backup);

    rc = sqlite3_errcode(db_);
    sqlite3_close(backup_db);

    return rc == SQLITE_OK;
}

// ============================================================================
// Helper Methods
// ============================================================================

std::vector<uint8_t> PersistentBackend::SerializeNode(const PatternNode& node) {
    std::ostringstream oss(std::ios::binary);
    node.Serialize(oss);
    std::string str = oss.str();
    return std::vector<uint8_t>(str.begin(), str.end());
}

PatternNode PersistentBackend::DeserializeNode(const std::vector<uint8_t>& blob) {
    std::string str(blob.begin(), blob.end());
    std::istringstream iss(str, std::ios::binary);
    return PatternNode::Deserialize(iss);
}

size_t PersistentBackend::GetDatabaseSize() const {
    struct stat st;
    if (stat(config_.db_path.c_str(), &st) == 0) {
        return st.st_size;
    }
    return 0;
}

void PersistentBackend::BeginTransaction() {
    ExecuteSQL("BEGIN TRANSACTION;");
}

void PersistentBackend::CommitTransaction() {
    ExecuteSQL("COMMIT;");
}

void PersistentBackend::RollbackTransaction() {
    ExecuteSQL("ROLLBACK;");
}

} // namespace dpan
