package com.example.smart_disease_prediction_sys;



import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.cardview.widget.CardView;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class RecentAssessmentAdapter extends RecyclerView.Adapter<RecentAssessmentAdapter.ViewHolder> {

    private List<RecentAssessment> assessmentList;

    public RecentAssessmentAdapter(List<RecentAssessment> assessmentList) {
        this.assessmentList = assessmentList;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_assessment, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        RecentAssessment assessment = assessmentList.get(position);

        holder.tvDate.setText(assessment.getDate());
        holder.tvType.setText(assessment.getType());
        holder.tvResult.setText(assessment.getResult());

        // Set background color based on result
        int color = ContextCompat.getColor(holder.itemView.getContext(), assessment.getColorRes());
        holder.cardView.setCardBackgroundColor(color);
    }

    @Override
    public int getItemCount() {
        return assessmentList.size();
    }

    public void setData(List<RecentAssessment> newData) {
        this.assessmentList = newData;
        notifyDataSetChanged();
    }

    public void clearData() {
        this.assessmentList.clear();
        notifyDataSetChanged();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        CardView cardView;
        TextView tvDate;
        TextView tvType;
        TextView tvResult;

        ViewHolder(@NonNull View itemView) {
            super(itemView);
            cardView = itemView.findViewById(R.id.cardAssessment);
            tvDate = itemView.findViewById(R.id.tvDate);
            tvType = itemView.findViewById(R.id.tvType);
            tvResult = itemView.findViewById(R.id.tvResult);
        }
    }
}