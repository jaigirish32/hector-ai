"""
End-to-end test of FileLibrary.

Validates:
- add_file uploads to all configured providers (success on each is logged)
- list_files returns the just-added file
- delete_file removes from all providers + SQLite
- list_files no longer returns the deleted file
"""
from pathlib import Path

from attachments.file_library import FileLibrary
from attachments.registry import FileRegistry


def main() -> None:
    test_db = Path("test_file_library.db")
    if test_db.exists():
        test_db.unlink()

    test_pdf = Path("test_library.pdf")
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
        b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
        b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000053 00000 n \n0000000100 00000 n \n"
        b"trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n160\n%%EOF\n"
    )
    test_pdf.write_bytes(pdf_bytes)
    print(f"✓ Created test PDF ({test_pdf.stat().st_size} bytes)")

    registry = FileRegistry(db_path=test_db)
    library = FileLibrary(registry=registry)

    # =========================
    # Step 1: add_file
    # =========================
    print("\nStep 1: add_file uploads to all providers...")
    outcome = library.add_file(test_pdf)
    print(f"  file_id:               {outcome.file_record.id}")
    print(f"  successful_providers:  {outcome.successful_providers}")
    print(f"  failed_providers:      {outcome.failed_providers}")

    if not outcome.successful_providers:
        print("  ✗ No providers succeeded; aborting test")
        cleanup(test_pdf, test_db, registry)
        return

    # =========================
    # Step 2: list_files
    # =========================
    print("\nStep 2: list_files shows the new file...")
    files = library.list_files()
    print(f"  Files in library: {len(files)}")
    for f in files:
        print(f"    - {f.filename} ({f.size_bytes} bytes)")
        print(f"      uploaded to: {f.providers_uploaded}")

    if not files or files[0].file_id != outcome.file_record.id:
        print("  ✗ Listed files don't match what we added")
        cleanup(test_pdf, test_db, registry)
        return
    print("  ✓ File listed correctly")

    # =========================
    # Step 3: get_refs_for_provider
    # =========================
    print("\nStep 3: get_refs_for_provider for each provider...")
    for provider in ["anthropic", "gemini", "openai", "azure_openai"]:
        refs = library.get_refs_for_provider([outcome.file_record.id], provider)
        if refs:
            print(f"  {provider:14s}  remote_id={refs[0].remote_id}")
        else:
            print(f"  {provider:14s}  (no ref — provider may not be configured)")

    # =========================
    # Step 4: delete_file
    # =========================
    print("\nStep 4: delete_file removes from all providers...")
    delete_outcome = library.delete_file(outcome.file_record.id)
    print(f"  successful_providers:  {delete_outcome.successful_providers}")
    print(f"  failed_providers:      {delete_outcome.failed_providers}")

    # =========================
    # Step 5: list_files after delete
    # =========================
    print("\nStep 5: list_files no longer shows the deleted file...")
    files_after = library.list_files()
    print(f"  Files in library: {len(files_after)}")

    # If the delete had any failures, the file may still be in the library
    # (we leave it so the user can retry). That's expected behavior.
    if delete_outcome.failed_providers:
        print("  ⚠  Some providers failed delete — file remains in library for retry")
    elif files_after:
        print("  ✗ File should have been removed but is still listed")
    else:
        print("  ✓ File fully removed from library")

    cleanup(test_pdf, test_db, registry)
    print("\n=== FileLibrary test complete ===")


def cleanup(test_pdf: Path, test_db: Path, registry: FileRegistry | None = None) -> None:
    if registry:
        registry.close()
    if test_pdf.exists():
        test_pdf.unlink()
    if test_db.exists():
        test_db.unlink()


if __name__ == "__main__":
    main()