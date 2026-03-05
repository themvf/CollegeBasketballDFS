import { redirect } from "next/navigation";

export default function SeasonBackfillPage() {
  redirect("/generate-lineup/backfill");
}
